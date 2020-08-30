import os

# Creates index.html file
try:
    index_file = open("index.html", "w")
    index_file.write('''<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>index</title>
        <link rel="stylesheet" type="text/css" href="css/style.css">
        <script type="text/javascript" src="js/ajax-utils.js"></script>
        <script type="text/javascript" src="js/script.js"></script>
    </head>
    <body>

        <h1>Title here</h1>
        
    </body>
    </html>
    ''')
    index_file.close()
except Exception:
    pass

# Creates directory for scripts
try:
    path = os.path.join(os.getcwd(), "js")
    os.mkdir(path)
except Exception:
    pass

# Creates script.js in js directory
try:
    js_file = open(os.path.join(path, "script.js"), "w")
    js_file.close()
except Exception:
    pass

# Creates ajax-utils.js in js directory
try:
    ajax_file = open(os.path.join(path, "ajax-utils.js"), "w")
    ajax_file.write('''(function (global) {


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
''')
    ajax_file.close()
except Exception:
    pass

# Creates directory for css
try:
    path = os.path.join(os.getcwd(), "css")
    os.mkdir(path)
except Exception:
    pass

# Creates style.css in css directory
try:
    css_file = open(os.path.join(path, "style.css"), "w")
    css_file.close()
except Exception:
    pass







