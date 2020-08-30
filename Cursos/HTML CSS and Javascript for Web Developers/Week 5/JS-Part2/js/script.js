// DOM manipulation

/*
console.log(document.getElementById("page_title")); // Gets element id
console.log(document instanceof HTMLDocument); // document is HTML
*/

// Event Handling

document.addEventListener("DOMContentLoaded",
	function (event) {

		function sayHello(event) {

			this.textContent = "Said it!";
			var name = document.getElementById("name").value;
			var message = "<h2>Hello " + name + "!</h2>"

			document.getElementById("message").innerHTML = message;

			if (name === "student") {
				var title = document.querySelector("#page_title").textContent;
				title += " & Lovin' it!";
				document.querySelector("#page_title").textContent = title;

			}
		}

		document.querySelector("body").addEventListener("mousemove", 
			function (event) {
				if (event.shiftKey === true) {
					console.log("x: " + event.clientX + "; " + "y: " +
						event.clientY);
				}
			}
		);

		document.querySelector("button").onclick = sayHello;

	}
);



