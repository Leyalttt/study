let a: unknown;
a = 123

let b: number;

// if (typeof a === 'number') {
//   b = a
// }

//断言, 等同于上面的写法
b = a as number
