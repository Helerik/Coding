
$( function () { // is JQuery, this is the same as document.addEventListener("DOMContentLoaded"...

	// Same as document.querySelector("#navbarToggle").addEventListener("blur"...
	$("#navbarToggle").blur(function(event){
		var screenWidth = window.innerWidth;
		if (screenWidth < 768) {
			$("#collapsable-nav").collapsable('hide');
		}
	});
});