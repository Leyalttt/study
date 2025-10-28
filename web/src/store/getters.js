//getters是处理state的
const getters = {
  //处理左侧列表的使用数量
  pageComponentTotalMap: (state)=> {
    //map->key: value形式 如:图文广告: 12
    let map = {}
    //存储小组件的key值
    let cName
    //拿到左侧组件列表
    const cList = state.pageData.componentList || []
    cList.forEach(item => {
      //用component是因为唯一
      cName = item.data.component  //组件的名称
      //判断对象[属性]没有值, 那么属性就不存在
      if(map[cName]){
        map[cName] += 1
      } else {
        map[cName] = 1
      }
    });
    return map
  }
}

//方便以后有其他getters好合并
export default Object.assign({}, getters)