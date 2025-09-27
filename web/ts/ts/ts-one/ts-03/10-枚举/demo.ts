//数字枚举
enum Color {
  red,
  yellow,
  blue
}

let r = Color.red
console.log(r); //0
let y = Color.yellow
console.log(y); //1

//字符串枚举
enum Gender {
  male = '男',
  female = '女'
}
let m = Gender.male
console.log(m)
let f = Gender.female
console.log(f)