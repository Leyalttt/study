let demo: any
demo = 123
demo = 'string'
demo = true

let abc
abc = 123
abc = 'yuyu'
abc = true

let str:string
str = abc
//使用 any 声明的变量它不仅影响自己，同时还影响别人
console.log('str' + ' ' + typeof str) //boolean
