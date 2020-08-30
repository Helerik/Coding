(function (global) {


	// set up a namespace for utility
	var ajaxUtils = {};


	// returns HTTP request object
	function getRequestObject() {
		if (window.XMLHttpRequest) {
			return (new XMLHttpRequest());
		}
		else if (window.ActiveXObject){
			// For old IE browsers
			return (new ActiveXObject("Microsoft.XMLHTTP"));
		}
		else {
			global.alert("Ajax is not supported!");
			return (null);
		}
	}

	// makes an Ajax GET request to 'requestUrl'
	ajaxUtils.sendGetRequest = 
	function (requestUrl, responseHandler, isJsonResponse) {
		var request = getRequestObject();
		request.onreadystatechange = function() {
			handleResponse(request, responseHandler, isJsonResponse);
		};
		request.open("GET", requestUrl, true);
		request.send(null); // for POST only
	};

	// executes handler if there are no errors
	function handleResponse(request, responseHandler, isJsonResponse) {
		if ((request.readyState == 4) && (request.status == 200)) {
			
			if (isJsonResponse == undefined) {
				isJsonResponse = true;
			}

			if (isJsonResponse) {
				// parse turns JSON string into JS object
				// stringify turns JS object into JSON string
				responseHandler(JSON.parse(request.responseText));
			}

			else {
				responseHandler(request.responseText);
			}
		}
	}

	global.$ajaxUtils = ajaxUtils;

})(window);