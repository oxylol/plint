// Plint · tiny touches. Nothing here is required.
// Greet by time of day in the tab title.
(function () {
  try {
    var h = new Date().getHours();
    var word = h < 5 ? "Night" : h < 12 ? "Morning" : h < 18 ? "Afternoon" : "Evening";
    document.title = "Plint · " + word;
  } catch (e) {}
})();
