/* TYPES IN JAVASCRIPT */

/* Primitive Types */

/* Boolean */

var truth_val = new Boolean(true);
var false_val = new Boolean(false);

/* undefined */

/* Never set a undefined value to a variable! */

/* Null */

var a = null;
console.log(a);

/* Number - always float type */

var a = new Number(10.5);
console.log(a);

/* ========================================================================= */

var x;
console.log(x);

if (x == undefined) {
	console.log("x is undefined")
}

x = 5;

if (x == undefined) {
	console.log("x is undefined")
}

else {
	console.log("x has been defined")
}

/* ========================================================================= */

/* Language Constructs */

/* String concatenation - just like python! */

var string = "Hello";
string += " World";
console.log(string + "!");

/* Math operations - very straight forward */

console.log((5 + 4) / 3 - 2*1);
console.log(undefined / 5);
console.log(10 / Infinity);

/* Truth and logic operators */

var x = 4, y = 4;
if (x == y) {
	console.log(x == y);
}

x = "4"; // language cohersion
if (x == y) {
	console.log(x == y);
}

if (x === y) {
	console.log("x is equal to y");
}
else {
	console.log("x is not equal to y");
}

/* What is considered false or true */

if (false || null || undefined || "" || 0 || NaN){
	console.log("This line never executes");
}
else {
	console.log("All false");
}

if (true && "non empty string" && 1 && -1 && "false"){
	console.log("All true");
}

// && is the and operator and || is the or operator //

/* Best practice for curly braces {} */

function A() // bad
{
	return 
	{
		name: "Erik"
	};
}

function B() { // better and sintatically more correct
	return {
		name: "Erik"
	};
}

console.log(A());
console.log(B());

/* For loop */

var sum = 0;
for (var i = 0; i < 10; i++) {
	sum += i;
}

console.log("Sum from 0 to 9 is: " + sum);

/* ========================================================================= */

/* Default values */

function orderChickenWith(sideDish) {
	sideDish = sideDish || "whatever";
	console.log("Chicken with " + sideDish);
}
orderChickenWith("noodles");

function orderChickenWith(sideDish = "whatever") { // just like python
	console.log("Chicken with " + sideDish);
}
orderChickenWith("noodles");

orderChickenWith();

/* ========================================================================= */

/* Creating objects */

var company = new Object();
company.name = "Facebook";
company.ceo = new Object();
company.ceo.firstName = "Mark";
company.ceo.favColor = "blue";

console.log(company);
console.log("The company's CEO is: " + company.ceo.firstName);
console.log("The company's name is: " + company["name"]);

/* company.stock price = 110; -> Invalid notation! */
company["stock price"] = 110; // This is similar to python dictionaries...
console.log(company);
console.log(company["stock price"]);

/* Better way of defining objects */
var Facebook = {
	name: "Facebook",
	ceo: {
		firstName: "Mark",
		lastName: "Zuckerberg",
		favColor: "blue"
	},
	"stock price": 110
};

console.log(Facebook);

