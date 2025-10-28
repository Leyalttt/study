<template>
  <div>
    <com-group 
      title="添加内容"
      tips="最多添加10条内容, 多条将会滚动展示, 拖动选中的内容对其进行排序"
      :name-block="true"
      >
    </com-group>
    <com-group  :bg-gray="true" :content-block="true">
      <pic-list 
        :image-list.sync="configData.noticelist" 
        add-place-holder="添加内容" 
        :show-pic="false"
      ></pic-list>
    </com-group>
    <com-group title="配置图表">
      <!-- 重置按钮 -->
      <el-button type="text" @click="configData.imageUrl = ''">重置</el-button>
      <up-load-box :image-url.sync="configData.imageUrl"></up-load-box>
    </com-group>

    <com-group title="背景颜色">
      <!-- 初始颜色不能为空, 在data中定义初始颜色 -->
      <el-button type="text" @click="configData.backgroundColor = initBgColor">重置</el-button>
      <el-color-picker v-model="configData.backgroundColor" size="small"></el-color-picker>
    </com-group>
    <com-group title="文字颜色">
      <!-- 初始颜色不能为空, 在data中定义初始颜色 -->
      <el-button type="text" @click="configData.textColor = initTxtColor">重置</el-button>
      <el-color-picker v-model="configData.textColor" size="small"></el-color-picker>
    </com-group>
  </div>
</template>

<script>
import ComGroup from'@/components/BasicUi/ComGroup'
import PicList from '../../BasicConfig/PicList.vue';
import UpLoadBox from '@/components/BasicUi/UpLoadBox'
export default {
  name: 'Notice',
  props: ['parmes'],
  data() {
    return {
      //当前组件的数据
      configData: JSON.parse(JSON.stringify(this.parmes)),
      //初始背景颜色
      initBgColor: '#FFF8E9',
      // 初始文本颜色
      initTxtColor: '#666666'
    }
  },
  watch: {
    //监听当前组件的数据变化
    configData: {
      //传递给父组件
      handler: function(newVal, oldVal) {
        console.log(newVal, 'newVal')
        this.$emit('editComponent', newVal)
      },
      deep: true
    }
  },
  components: {
    ComGroup,
    PicList,
    UpLoadBox
  },

}
</script>

<style lang="less" scoped>

</style>
