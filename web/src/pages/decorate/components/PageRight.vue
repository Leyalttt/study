<template>
  <div class="page-right">
    <change-set-type v-model="setType"/>
    <!-- 展示页面设置, 根据settype值进行判断 -->
    <div v-show="setType === 1">
      <set-page-info />
    </div>
    <!-- 展示组件设置, 根据settype值进行判断 -->
    <div v-show="setType === 2">
      <com-title :title="currentComponent && currentComponent.name || '组件管理'"></com-title>
      <!-- 生效时间 -->
      <com-valid-time v-if="currentComponent" v-model="currentComponent.data.validTime"></com-valid-time>
      <!-- 组件里面的数据component 
        parmes是传递给组件的数据
        传递自定义事件接受子组件传递传递过来的数据, 并更新store中的pageData
        v-if="currentComponent" 为了容错, 
        要给组件加上sort 用于排序
      -->
      <component 
        :is="currentComponent.data.component"
        v-if="currentComponent"
        :parmes="currentComponent.data"
        @editComponent = "editComponent"
        >
        </component>

    </div>

  </div>
</template>

<script>
import ChangeSetType from './components/ChangeSetType/index.vue'
import SetPageInfo from './components/SetPageInfo'
import {dynamicComponents} from '@/utils'
import ComValidTime from '@/components/BasicUi/ComValidTime'
import ComTitle from '@/components/BasicUi/ComTitle'
import {mapState, mapActions} from'vuex'
import NoSelect from './components/NoSelect'
export default {
  name: 'PageRight',
  // 动态注册所有可配置组件
  components: {
    ChangeSetType,
    SetPageInfo,
    //解构后相当于引入组件
    ...dynamicComponents(),
    ComValidTime,
    ComTitle,
    NoSelect
  },
  data() {
    return {
    }
  },
  computed: {
    ...mapState(['pageData', 'activeComponentId']),
    //拿到store->setType->然后传递给子组件
    //通过set,get去修改computed的值
    setType: {
      get() {
        return this.$store.state.setType
      },
      //调用mutations->SET_SETTYPE修改state里面的setType
      set(val) {
        this.$store.commit("SET_SETTYPE", val)
      },
      
      
    },
    // 找到当前组件数据
    currentComponent() {
        // 组件列表
        const componentList = this.pageData.componentList
        //判断componentList 有值, 并且长度大于0, 满足条件去找到item.id与当前选中的组件id进行比较, 否组为空
        return componentList && componentList.length > 0 ? componentList.find(item=> item.id === this.activeComponentId) : null
      }
  },
  methods: {
    ...mapActions(['pageChange']),
    //接受子组件回传的数据并更新store中的pageData
    editComponent(data) {
      this.pageChange({
        type: 'edit',
        id: this.activeComponentId,
        //子组件传递过来的newVal
        data
      })
    }
  },
}
</script>

<style lang="less">
@import url("~@/styles/icon.less");

.page-right {
  position: absolute;
  top: 56px;
  right: 0;
  bottom: 0;
  width: 376px;
  padding-bottom: 50px;
  overflow-x: hidden;
  overflow-y: auto;
  background: #fff;
}

// 组件配置模块
.com-group {
  // slide 滑块
  .el-slider {
    width: 95%;
    .el-slider__input {
      width: 60px;
    }
    .el-slider__runway {
      margin-right: 80px;
    }
  }
  // 单选框
  .el-radio {
    line-height: 30px;
    color: #323233;
  }
}

// 右上角删除样式
.up-pic-item-delete {
  position: absolute;
  cursor: pointer;
  font-size: 20px;
  right: -10px;
  top: -10px;
  color: #bbb;
  background: #fff;
  border-radius: 50%;
  visibility: hidden;
  z-index: 2;
  &:hover {
    color: #aaa;
  }
}
</style>
