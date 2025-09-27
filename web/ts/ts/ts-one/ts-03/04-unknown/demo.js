var un;
un = 123;
un = 'unun';
un = true;
var c;
c = 'ccc';
var d;
d = c; //此时 TS 解析器提示是报错的
//虽然 变量中的字面量都是 string，但是 d 是 string 类型 c 是 unknown 所以不能赋值
//假如我就想让 c 的值赋值给 d,该怎么操作呢？
if (typeof c === 'string') {
    d = c;
}
//unknown 看成安全的any类型
