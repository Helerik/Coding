/* Arrays */

var array = new Array();
array[0] = "Erik";
array[1] = 2;
array[2] = function(name) {
	console.log("Hello " + name);
};
array[3] = {course: "HTML, CSS & JS"};

console.log(array);
console.log(array[0]);
console.log(array[1]);
console.log(array[2]("Erik"));
console.log(array[3].course);

var names = ["Erik", "Amanda", "Glaucia"]; // just like in python!
console.log(names);

for (var i = 0; i < names.length; i++) {
	// Generates an index
	console.log("Hello " + names[i]);
}

for (var name of names) { // almost? same as python
	// Gets the name itself
	console.log("Hello " + name);
}

for (var name in names) {
	// Gets index of name and properties!
	console.log("Hello " + names[name]);
}

/* ========================================================================= */

/* Closures */

function makeMultiplier (multiplier) {
	return (
		function (x) {
			return multiplier * x;
		}
	);
}

var doubleAll = makeMultiplier(2);
console.log(doubleAll(10));

/* ========================================================================= */

/* Immediately Invoked Function Expression - IIFE */

(function (name) {
	console.log("Hello " + name + "!");
})("World")