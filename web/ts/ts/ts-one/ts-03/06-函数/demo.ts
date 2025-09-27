//泛型
//函数后面的是返回值类型
function get(name: string): number {
  return 123
}
get('张三')

//返回值void
function fun(): void {
  //报错
  // return ''
  // return 123
  // return true
  //不报错
  return null
  return undefined
}

//返回值never
function func(): never {
  //抛出异常
  throw new Error('出现错误了')
}