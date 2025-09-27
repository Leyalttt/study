//数字枚举
var Color;
(function (Color) {
    Color[Color["red"] = 0] = "red";
    Color[Color["yellow"] = 1] = "yellow";
    Color[Color["blue"] = 2] = "blue";
})(Color || (Color = {}));
var r = Color.red;
console.log(r); //0
var y = Color.yellow;
console.log(y); //1
//字符串枚举
var Gender;
(function (Gender) {
    Gender["male"] = "\u7537";
    Gender["female"] = "\u5973";
})(Gender || (Gender = {}));
var m = Gender.male;
console.log(m);
var f = Gender.female;
console.log(f);
