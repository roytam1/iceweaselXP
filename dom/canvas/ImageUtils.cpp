/* -*- Mode: C++; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim:set ts=2 sw=2 sts=2 et cindent: */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "ImageUtils.h"
#include "ImageBitmapUtils.h"
#include "ImageContainer.h"
#include "mozilla/AlreadyAddRefed.h"
#include "mozilla/dom/ImageBitmapBinding.h"
#include "mozilla/ErrorResult.h"

#if defined(TT_MEMUTIL) && defined(_MSC_VER)
#include <omp.h>
#endif

#include <tmmintrin.h>
#include "mozilla/SSE.h"
#include "gfxUtils.h"

using namespace mozilla::layers;
using namespace mozilla::gfx;

namespace mozilla {
namespace dom {

static ImageBitmapFormat
GetImageBitmapFormatFromSurfaceFromat(SurfaceFormat aSurfaceFormat)
{
  switch (aSurfaceFormat) {
  case SurfaceFormat::B8G8R8A8:
  case SurfaceFormat::B8G8R8X8:
    return ImageBitmapFormat::BGRA32;
  case SurfaceFormat::R8G8B8A8:
  case SurfaceFormat::R8G8B8X8:
    return ImageBitmapFormat::RGBA32;
  case SurfaceFormat::R8G8B8:
    return ImageBitmapFormat::RGB24;
  case SurfaceFormat::B8G8R8:
    return ImageBitmapFormat::BGR24;
  case SurfaceFormat::HSV:
    return ImageBitmapFormat::HSV;
  case SurfaceFormat::Lab:
    return ImageBitmapFormat::Lab;
  case SurfaceFormat::Depth:
    return ImageBitmapFormat::DEPTH;
  case SurfaceFormat::A8:
    return ImageBitmapFormat::GRAY8;
  case SurfaceFormat::R5G6B5_UINT16:
  case SurfaceFormat::YUV:
  case SurfaceFormat::NV12:
  case SurfaceFormat::UNKNOWN:
  default:
    return ImageBitmapFormat::EndGuard_;
  }
}

static ImageBitmapFormat
GetImageBitmapFormatFromPlanarYCbCrData(layers::PlanarYCbCrData const *aData)
{
  MOZ_ASSERT(aData);

  if (aData->mYSkip == 0 && aData->mCbSkip == 0 && aData->mCrSkip == 0) { // Possibly three planes.
    if (aData->mCbChannel >= aData->mYChannel + aData->mYSize.height * aData->mYStride &&
        aData->mCrChannel >= aData->mCbChannel + aData->mCbCrSize.height * aData->mCbCrStride) { // Three planes.
      if (aData->mYSize.height == aData->mCbCrSize.height) {
        if (aData->mYSize.width == aData->mCbCrSize.width) {
          return ImageBitmapFormat::YUV444P;
        } else if (((aData->mYSize.width + 1) / 2) == aData->mCbCrSize.width) {
          return ImageBitmapFormat::YUV422P;
        }
      } else if (((aData->mYSize.height + 1) / 2) == aData->mCbCrSize.height) {
        if (((aData->mYSize.width + 1) / 2) == aData->mCbCrSize.width) {
          return ImageBitmapFormat::YUV420P;
        }
      }
    }
  } else if (aData->mYSkip == 0 && aData->mCbSkip == 1 && aData->mCrSkip == 1) { // Possibly two planes.
    if (aData->mCbChannel >= aData->mYChannel + aData->mYSize.height * aData->mYStride &&
        aData->mCbChannel == aData->mCrChannel - 1) { // Two planes.
      if (((aData->mYSize.height + 1) / 2) == aData->mCbCrSize.height &&
          ((aData->mYSize.width + 1) / 2) == aData->mCbCrSize.width) {
        return ImageBitmapFormat::YUV420SP_NV12;  // Y-Cb-Cr
      }
    } else if (aData->mCrChannel >= aData->mYChannel + aData->mYSize.height * aData->mYStride &&
               aData->mCrChannel == aData->mCbChannel - 1) { // Two planes.
      if (((aData->mYSize.height + 1) / 2) == aData->mCbCrSize.height &&
          ((aData->mYSize.width + 1) / 2) == aData->mCbCrSize.width) {
        return ImageBitmapFormat::YUV420SP_NV21;  // Y-Cr-Cb
      }
    }
  }

  return ImageBitmapFormat::EndGuard_;
}

// ImageUtils::Impl implements the _generic_ algorithm which always readback
// data in RGBA format into CPU memory.
// Since layers::CairoImage is just a warpper to a DataSourceSurface, the
// implementation of CairoSurfaceImpl is nothing different to the generic
// version.
class ImageUtils::Impl
{
public:
  explicit Impl(layers::Image* aImage)
  : mImage(aImage)
  , mSurface(nullptr)
  {
  }

  virtual ~Impl() = default;

  virtual ImageBitmapFormat
  GetFormat() const
  {
    return GetImageBitmapFormatFromSurfaceFromat(Surface()->GetFormat());
  }

  virtual uint32_t
  GetBufferLength() const
  {
    const uint32_t stride = Surface()->Stride();
    const IntSize size = Surface()->GetSize();
    return (uint32_t)(size.height * stride);
  }

  virtual UniquePtr<ImagePixelLayout>
  MapDataInto(uint8_t* aBuffer,
              uint32_t aOffset,
              uint32_t aBufferLength,
              ImageBitmapFormat aFormat,
              ErrorResult& aRv) const
  {
    DataSourceSurface::ScopedMap map(Surface(), DataSourceSurface::READ);
    if (!map.IsMapped()) {
      aRv.Throw(NS_ERROR_ILLEGAL_VALUE);
      return nullptr;
    }

    // Copy or convert data.
    UniquePtr<ImagePixelLayout> srcLayout =
      CreateDefaultPixelLayout(GetFormat(), Surface()->GetSize().width,
                               Surface()->GetSize().height, map.GetStride());

    // Prepare destination buffer.
    uint8_t* dstBuffer = aBuffer + aOffset;
    UniquePtr<ImagePixelLayout> dstLayout =
      CopyAndConvertImageData(GetFormat(), map.GetData(), srcLayout.get(),
                              aFormat, dstBuffer);

    if (!dstLayout) {
      aRv.Throw(NS_ERROR_NOT_AVAILABLE);
      return nullptr;
    }

    return dstLayout;
  }

protected:
  Impl() {}

  DataSourceSurface* Surface() const
  {
    if (!mSurface) {
      RefPtr<SourceSurface> surface = mImage->GetAsSourceSurface();
      MOZ_ASSERT(surface);

      mSurface = surface->GetDataSurface();
      MOZ_ASSERT(mSurface);
    }

    return mSurface.get();
  }

  RefPtr<layers::Image> mImage;
  mutable RefPtr<DataSourceSurface> mSurface;
};

// YUVImpl is optimized for the layers::PlanarYCbCrImage and layers::NVImage.
// This implementation does not readback data in RGBA format but keep it in YUV
// format family.
class YUVImpl final : public ImageUtils::Impl
{
public:
  explicit YUVImpl(layers::Image* aImage)
  : Impl(aImage)
  {
    MOZ_ASSERT(aImage->GetFormat() == ImageFormat::PLANAR_YCBCR ||
               aImage->GetFormat() == ImageFormat::NV_IMAGE);
  }

  ImageBitmapFormat GetFormat() const override
  {
    return GetImageBitmapFormatFromPlanarYCbCrData(GetPlanarYCbCrData());
  }

  uint32_t GetBufferLength() const override
  {
    if (mImage->GetFormat() == ImageFormat::PLANAR_YCBCR) {
      return mImage->AsPlanarYCbCrImage()->GetDataSize();
    } else {
      return mImage->AsNVImage()->GetBufferSize();
    }
  }

  UniquePtr<ImagePixelLayout>
  MapDataInto(uint8_t* aBuffer,
              uint32_t aOffset,
              uint32_t aBufferLength,
              ImageBitmapFormat aFormat,
              ErrorResult& aRv) const override
  {
    // Prepare source buffer and pixel layout.
    const PlanarYCbCrData* data = GetPlanarYCbCrData();

    UniquePtr<ImagePixelLayout> srcLayout =
      CreatePixelLayoutFromPlanarYCbCrData(data);

    // Do conversion.
    UniquePtr<ImagePixelLayout> dstLayout =
      CopyAndConvertImageData(GetFormat(), data->mYChannel, srcLayout.get(),
                              aFormat, aBuffer+aOffset);

    if (!dstLayout) {
      aRv.Throw(NS_ERROR_NOT_AVAILABLE);
      return nullptr;
    }

    return dstLayout;
  }

private:
  const PlanarYCbCrData* GetPlanarYCbCrData() const
  {
    if (mImage->GetFormat() == ImageFormat::PLANAR_YCBCR) {
      return mImage->AsPlanarYCbCrImage()->GetData();
    } else {
      return mImage->AsNVImage()->GetData();
    }
  }
};

// TODO: optimize for other platforms.
// For GONK: implement GrallocImageImpl, GrallocPlanarYCbCrImpl and GonkCameraImpl.
// For Windows: implement D3D9RGB32TextureImpl and D3D11ShareHandleTextureImpl.
// Others: SharedBGRImpl, MACIOSrufaceImpl, GLImageImpl, SurfaceTextureImpl
//         EGLImageImpl and OverlayImegImpl.

ImageUtils::ImageUtils(layers::Image* aImage)
: mImpl(nullptr)
{
  MOZ_ASSERT(aImage, "Create ImageUtils with nullptr.");
  switch(aImage->GetFormat()) {
  case mozilla::ImageFormat::PLANAR_YCBCR:
  case mozilla::ImageFormat::NV_IMAGE:
    mImpl = new YUVImpl(aImage);
    break;
  case mozilla::ImageFormat::CAIRO_SURFACE:
  default:
    mImpl = new Impl(aImage);
  }
}

ImageUtils::~ImageUtils()
{
  if (mImpl) { delete mImpl; mImpl = nullptr; }
}

ImageBitmapFormat
ImageUtils::GetFormat() const
{
  MOZ_ASSERT(mImpl);
  return mImpl->GetFormat();
}

uint32_t
ImageUtils::GetBufferLength() const
{
  MOZ_ASSERT(mImpl);
  return mImpl->GetBufferLength();
}

UniquePtr<ImagePixelLayout>
ImageUtils::MapDataInto(uint8_t* aBuffer,
                        uint32_t aOffset,
                        uint32_t aBufferLength,
                        ImageBitmapFormat aFormat,
                        ErrorResult& aRv) const
{
  MOZ_ASSERT(mImpl);
  MOZ_ASSERT(aBuffer, "Map data into a null buffer.");
  return mImpl->MapDataInto(aBuffer, aOffset, aBufferLength, aFormat, aRv);
}

} // namespace dom


void
GetImageData_component(uint8_t* _src, uint8_t* _dst,
                       int32_t width, int32_t height,
                       uint32_t srcStride, uint32_t dstStride)
{
    uint8_t *srcFirst = _src;
    uint8_t *dstFirst = _dst;

#if defined(TT_MEMUTIL) && defined(_MSC_VER)
    int omp_thread_counts = omp_get_max_threads();

#pragma omp parallel for schedule(guided) default(none) \
shared(srcFirst, dstFirst, width, height, srcStride, dstStride, gfxUtils::sUnpremultiplyTable) \
if (omp_thread_counts >= 2 && \
    height >= (int32_t)omp_thread_counts && \
    width * height >= 4096)
#endif // defined(TT_MEMUTIL) && defined(_MSC_VER)
    for (int32_t j = 0; j < height; ++j) {
        uint8_t *src = srcFirst + (srcStride * j);
        uint8_t *dst = dstFirst + (dstStride * j);

        for (int32_t i = 0; i < width; ++i) {
            // XXX Is there some useful swizzle MMX we can use here?
#ifdef IS_LITTLE_ENDIAN
            uint8_t b = *src++;
            uint8_t g = *src++;
            uint8_t r = *src++;
            uint8_t a = *src++;
#else
            uint8_t a = *src++;
            uint8_t r = *src++;
            uint8_t g = *src++;
            uint8_t b = *src++;
#endif
            // Convert to non-premultiplied color
            *dst++ = gfxUtils::sUnpremultiplyTable[a * 256 + r];
            *dst++ = gfxUtils::sUnpremultiplyTable[a * 256 + g];
            *dst++ = gfxUtils::sUnpremultiplyTable[a * 256 + b];
            *dst++ = a;
        }
    }
}

void
PutImageData_component(uint8_t* _src, uint8_t* _dst,
                       int32_t width, int32_t height,
                       uint32_t srcStride, uint32_t dstStride, uint8_t alphaMask)
{
    uint8_t *srcFirst = _src;
    uint8_t *dstFirst = _dst;

    if (mozilla::supports_ssse3()) {
        static const __m128i msk_alpha = _mm_set1_epi32(0xFF000000);
        static const __m128i alphaMask128 = _mm_set1_epi32(((uint32_t)alphaMask) << 24);
        static const __m128i sfl_alphaLo = _mm_set_epi8((char)0x80, 7, (char)0x80, 7, (char)0x80, 7, (char)0x80, 7, (char)0x80, 3, (char)0x80, 3, (char)0x80, 3, (char)0x80, 3);
        static const __m128i sfl_alphaHi = _mm_set_epi8((char)0x80, 15, (char)0x80, 15, (char)0x80, 15, (char)0x80, 15, (char)0x80, 11, (char)0x80, 11, (char)0x80, 11, (char)0x80, 11);
        static const __m128i word_add = _mm_set1_epi16(0x00FF);
        static const __m128i word_mul = _mm_set_epi16(0, 257, 257, 257, 0, 257, 257, 257);
        static const __m128i sfl_bgra = _mm_set_epi8(15, 12, 13, 14, 11, 8, 9, 10, 7, 4, 5, 6, 3, 0, 1, 2);

#if defined(TT_MEMUTIL) && defined(_MSC_VER)
        int omp_thread_counts = omp_get_max_threads();

#pragma omp parallel for schedule(guided) default(none) \
shared(srcFirst, dstFirst, width, height, srcStride, dstStride, gfxUtils::sPremultiplyTable) \
if (omp_thread_counts >= 2 && \
    height >= (int32_t)omp_thread_counts && \
    width * height >= 12000)
#endif // defined(TT_MEMUTIL) && defined(_MSC_VER)
        for (int j = 0; j < height; j++) {
            uint8_t *src = srcFirst + (srcStride * j);
            uint8_t *dst = dstFirst + (dstStride * j);
            int32_t i = width;

            while (i >= 1 && ((uintptr_t)dst & 15)) {
                uint8_t r = *src++;
                uint8_t g = *src++;
                uint8_t b = *src++;
                uint8_t a = *src++;

                // Convert to premultiplied color (losslessly if the input came from getImageData)
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + b];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + g];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + r];
                *dst++ = a | alphaMask;
                i -= 1;
            }

            const int srcMissalignedBytes = ((uintptr_t)src & 15);

            if (srcMissalignedBytes == 0) {
                while (i >= 4) {
                    __m128i xmb = _mm_load_si128((__m128i*)src);
                    __m128i xmwLo = _mm_unpacklo_epi8(xmb, _mm_setzero_si128());
                    __m128i xmwHi = _mm_unpackhi_epi8(xmb, _mm_setzero_si128());

                    __m128i xmwAlpha = _mm_and_si128(xmb, msk_alpha);
                    xmwAlpha = _mm_or_si128(xmwAlpha, alphaMask128);

                    __m128i xmwAlphaLo = _mm_shuffle_epi8(xmb, sfl_alphaLo);
                    __m128i xmwAlphaHi = _mm_shuffle_epi8(xmb, sfl_alphaHi);

                    xmwLo = _mm_mullo_epi16(xmwLo, xmwAlphaLo);
                    xmwLo = _mm_adds_epu16(xmwLo, word_add);
                    xmwLo = _mm_mulhi_epu16(xmwLo, word_mul);

                    xmwHi = _mm_mullo_epi16(xmwHi, xmwAlphaHi);
                    xmwHi = _mm_adds_epu16(xmwHi, word_add);
                    xmwHi = _mm_mulhi_epu16(xmwHi, word_mul);

                    __m128i xmRes = _mm_packus_epi16(xmwLo, xmwHi);
                    xmRes = _mm_or_si128(xmRes, xmwAlpha);
                    xmRes = _mm_shuffle_epi8(xmRes, sfl_bgra);
                    _mm_store_si128((__m128i*)dst, xmRes);

                    src += 16;
                    dst += 16;
                    i -= 4;
                }
            } else {
                __m128i xmLoadPre = _mm_load_si128((__m128i*)(src - srcMissalignedBytes));

                while (i >= 4) {
                    __m128i xmLoadNext = _mm_load_si128((__m128i*)(src - srcMissalignedBytes + 16));
                    __m128i xmb;

                    switch (srcMissalignedBytes) {
                    case 1:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 1);
                        break;
                    case 2:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 2);
                        break;
                    case 3:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 3);
                        break;
                    case 4:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 4);
                        break;
                    case 5:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 5);
                        break;
                    case 6:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 6);
                        break;
                    case 7:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 7);
                        break;
                    case 8:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 8);
                        break;
                    case 9:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 9);
                        break;
                    case 10:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 10);
                        break;
                    case 11:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 11);
                        break;
                    case 12:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 12);
                        break;
                    case 13:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 13);
                        break;
                    case 14:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 14);
                        break;
                    case 15:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 15);
                        break;
                    }
                    xmLoadPre = xmLoadNext;

                    __m128i xmwLo = _mm_unpacklo_epi8(xmb, _mm_setzero_si128());
                    __m128i xmwHi = _mm_unpackhi_epi8(xmb, _mm_setzero_si128());

                    __m128i xmwAlpha = _mm_and_si128(xmb, msk_alpha);
                    xmwAlpha = _mm_or_si128(xmwAlpha, alphaMask128);

                    __m128i xmwAlphaLo = _mm_shuffle_epi8(xmb, sfl_alphaLo);
                    __m128i xmwAlphaHi = _mm_shuffle_epi8(xmb, sfl_alphaHi);

                    xmwLo = _mm_mullo_epi16(xmwLo, xmwAlphaLo);
                    xmwLo = _mm_adds_epu16(xmwLo, word_add);
                    xmwLo = _mm_mulhi_epu16(xmwLo, word_mul);

                    xmwHi = _mm_mullo_epi16(xmwHi, xmwAlphaHi);
                    xmwHi = _mm_adds_epu16(xmwHi, word_add);
                    xmwHi = _mm_mulhi_epu16(xmwHi, word_mul);

                    __m128i xmRes = _mm_packus_epi16(xmwLo, xmwHi);
                    xmRes = _mm_or_si128(xmRes, xmwAlpha);
                    xmRes = _mm_shuffle_epi8(xmRes, sfl_bgra);
                    _mm_store_si128((__m128i*)dst, xmRes);

                    src += 16;
                    dst += 16;
                    i -= 4;
                }
            }

            while (i >= 1) {
                uint8_t r = *src++;
                uint8_t g = *src++;
                uint8_t b = *src++;
                uint8_t a = *src++;

                // Convert to premultiplied color (losslessly if the input came from getImageData)
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + b];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + g];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + r];
                *dst++ = a | alphaMask;
                i -= 1;
            }
        }
    } else {
#if defined(TT_MEMUTIL) && defined(_MSC_VER)
        int omp_thread_counts = omp_get_max_threads();

#pragma omp parallel for schedule(guided) default(none) \
shared(srcFirst, dstFirst, width, height, srcStride, dstStride, gfxUtils::sPremultiplyTable) \
if (omp_thread_counts >= 2 && \
    height >= (int32_t)omp_thread_counts && \
    width * height >= 4096)
#endif // defined(TT_MEMUTIL) && defined(_MSC_VER)
        for (int64_t j = 0; j < height; j++) {
            uint8_t *src = srcFirst + (srcStride * j);
            uint8_t *dst = dstFirst + (dstStride * j);

            for (int32_t i = 0; i < width; i++) {
                // XXX Is there some useful swizzle MMX we can use here?
                uint8_t r = *src++;
                uint8_t g = *src++;
                uint8_t b = *src++;
                uint8_t a = *src++;

                // Convert to premultiplied color (losslessly if the input came from getImageData)
#if MOZ_LITTLE_ENDIAN
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + b];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + g];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + r];
                *dst++ = a | alphaMask;
#else
                *dst++ = a | alphaMask;
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + r];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + g];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + b];
#endif
            }
        }
    }
}


} // namespace mozilla
