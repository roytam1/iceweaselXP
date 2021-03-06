function test() {
  waitForExplicitFinish();

  var tab = gBrowser.addTab();

  tab.addEventListener("TabClose", function() {
    tab.removeEventListener("TabClose", arguments.callee);

    ok(tab.linkedBrowser, "linkedBrowser should still exist during the TabClose event");

    executeSoon(function() {
      ok(!tab.linkedBrowser, "linkedBrowser should be gone after the TabClose event");

      finish();
    });
  });

  gBrowser.removeTab(tab);
}
