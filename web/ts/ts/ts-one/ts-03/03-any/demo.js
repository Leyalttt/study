var demo;
demo = 123;
demo = 'string';
demo = true;
var abc;
abc = 123;
abc = 'yuyu';
abc = true;
var str;
str = abc;
//使用 any 声明的变量它不仅影响自己，同时还影响别人
console.log('str' + ' ' + typeof str); //str boolean
