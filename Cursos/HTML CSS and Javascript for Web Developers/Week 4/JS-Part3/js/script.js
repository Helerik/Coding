function test() {
	console.log(this);
	this.myName = "Erik";
}
test();
console.log(window.myName);

/* Function Constructor and Prototype*/

/* 
function Circle (radius) {
	this.radius = radius;
	this.getArea = // Badness! Makes the getArea be created everytime a new Circle object is created
		function () {
			return Math.PI * Math.pow(this.radius, 2)
		};
}
*/

function Circle (radius) {
	this.radius = radius;
}
Circle.prototype.getArea = 
	function () {
		return Math.PI * Math.pow(this.radius, 2)
	}

var myCircle = new Circle(10);
var myOtherCircle = new Circle(20);
console.log(myCircle);
console.log(myOtherCircle);

console.log(myCircle.getArea());

/* ========================================================================= */

/* Object literals and this */

var literalCircle = {
	radius: 10,

	getArea: function() {
		var self = this;
		console.log(this);

		var increaseRadius = function() {
			self.radius = 20;
		};
		increaseRadius();

		return Math.PI * Math.pow(this.radius, 2)
	}
};

console.log(literalCircle.getArea());
