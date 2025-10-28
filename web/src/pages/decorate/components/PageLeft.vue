<template>
  <div class="page-left">
    <el-collapse v-model="activeNames">
      <!-- name	唯一标志符 -->
      <!-- el-collapse-item 是小组件 -->
      <el-collapse-item 
        v-for="(item, index) in componentlist"
        :key="index"
        :title="item.title"
        class="component-item"
        :name="index + 1"
      >
      <!-- 事件开始拖拽 , 需要知道拖拽的是谁, 并且次数加1
       使用了多少次存在vuex中
       @dragstart 事件在用户开始拖动元素或被选择的文本时调用
       @dragend 事件在拖放操作结束时触发-->
        <ul class="component-list">
          <li 
            :class="draggableEnable(component)? 'drag-enabled': 'drag-disabled'"
            :draggable="draggableEnable(component)"
            @dragstart="onDragStart(component)"
            @dragend="onDragEnd"
          
            v-for="(component, size) in item.components"
            :key="size"
          >
            <i :class="component.iconClass" style="font-size: 28px"></i>
            <p class="name">{{component.name}}</p>
            <!-- 进行容错 -->
            <p class="num"> {{ componentMap[component.data.component] || 0}}/{{component.maxNumForAdd}}</p>
          </li>
        </ul>
      </el-collapse-item>
    </el-collapse>
  </div>
</template>

<script>
import componentlist from '@/config/component-list';
import {mapActions, mapMutations, mapState} from 'vuex'

export default {
  data() {
    return {
      activeNames: [],
      componentlist, 
    }
  },
  computed:{
    ...mapState(['addComponentIndex', 'dragComponent']),
    //页面组件的被使用数量 key:value形式
    componentMap() {
      return this.$store.getters.pageComponentTotalMap
    }
  },
  mounted() {
    console.log(this.component, 'huhu')
  },
  methods: {
    //解构两个方法
    ...mapMutations(['SET_DRAG_STATE', 'SET_DRAG_COMPONENT', 'VIEW_SET_ACTIVE', 'SET_DRAG_INDEX']),
    ...mapActions(['pageChange']),
    
    //控制当前元素是否可以拖拽
    draggableEnable(component) {
      //当前组件被使用的次数与组件被使用次数的上线进行比较
      //被使用的次数从store中获取, 可以再computed中定义函数, 也可以用辅助函数map
      let curNum = this.componentMap[component.data.component] || 0 
      // console.log(curNum)
      return curNum < component.maxNumForAdd
    },
    //拖拽开始
    onDragStart(component) {
      // console.log(component, 'onDragStart')
      // console.log(JSON.parse(JSON.stringify(component)), 'json')
      //将其状态改为true, 表示拖拽开始
      this.SET_DRAG_STATE(true)
      //component 是引用数据类型, 对其进行深拷贝, 避免在state中修改源数据
      this.SET_DRAG_COMPONENT(JSON.parse(JSON.stringify(component)))
    },
    //拖拽结束
    onDragEnd() {
      //将是否可拖拽改成不可拖拽
    this.SET_DRAG_STATE(false)
    //获取拖动组件要添加的位置
    let addIndex = this.addComponentIndex
    //将拖拽组件的数据传递给跨源通信
    if (addIndex != null) {
      this.pageChange({
      type: 'add',
      //添加的索引位置
      index: addIndex,
      //拖拽的组件数据
      data: this.dragComponent
    })
      this.SET_DRAG_INDEX(null)
        // console.log(addIndex, 'addIndex')
      this.VIEW_SET_ACTIVE(addIndex)
    }
    

      

    }
  },
  
  
}
</script>

<style lang="less" scoped>
// 左侧框架
.page-left {
  position: absolute;
  top: 56px;
  left: 0;
  width: 186px;
  overflow-x: hidden;
  overflow-y: auto;
  bottom: 0;
  background: #fff;
  user-select: none;
}

// 组件列表
.component-item {
  padding: 0 20px;
  margin-top: 22px;
  .component-list {
    overflow: hidden;
    li {
      float: left;
      width: 50%;
      font-size: 12px;
      padding-bottom: 8px;
      text-align: center;
      &.drag-enabled{
        cursor: move; //被悬浮的物体可被移动
      }
      &.drag-disabled{
        cursor: not-allowed; //不能执行
      }
      &.drag-enabled:hover {
        background: @color-1;
        color: #fff;
        border-radius: 2px;
        .ico {
          background-position: 0 -32px;
        }
        .num, .name{
          color: #ffffff !important;
        }
      }
      .ico {
        display: inline-block;
        margin-top: 8px;
        background-position: 0 0;
        height: 32px;
        width: 32px;
        background-size: cover;
      }
      .name {
        line-height: 16px;
        margin-top: -4px;
      }
      .num{
        font-size: 12px;
        color: #999999;
      }
    }
  }
}

// 折叠面板样式
.el-collapse {
  border: none;
  /deep/ .el-collapse-item__header {
    border: none;
    height: 30px;
    line-height: 30px;
  }
  /deep/ .el-collapse-item__wrap {
    border: none;
  }
  /deep/ .el-collapse-item__content {
    padding-bottom: 0;
  }
}
</style>