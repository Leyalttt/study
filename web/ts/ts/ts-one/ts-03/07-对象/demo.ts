//以下都是对象类型
let obj: Object
obj = {}
obj = []
obj = function() {

}

//声明一个对象类型, 里面的name属性是string类型, age是number类型
let object: {name: string, age: number}
// object = {name: '张三', age: 18}
// 少写一个属性会报错
// object = {name: '张三'}

//声明一个对象类型, 里面的name属性是string类型, age是number类型或者没有
let person: {name: string, age?: number}
person = {name: '张三'}
person = {name: '张三', age: 18}


//属性名string类型, 属性值unknown类型
let animal: {name: string, [propName: string]:unknown}
animal = {name: "大象", gender: 'male'}
animal = {name: "大象", age: 18, gender: 'male'}
