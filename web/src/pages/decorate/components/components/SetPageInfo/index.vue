<template>
  <div>
    <!-- v-model和store中的state页面初始化星星双向绑定 -->
    <com-title title="页面设置" />
    <com-group title="页面名称" name-bock content-block>
      <el-input 
      v-model.lazy="name" 
      placeholder="请输入用户名称"></el-input>
    </com-group>

    <!-- 页面描述: 微信分享文案 .lazy change时才会触发 -->
    <com-group 
      title="微信分享文案" name-bock content-block
      >
      <el-input 
        v-model.lazy="shareDesc" 
        class="input-name"
        maxlength="28" 
        placeholder="用户通过微信分享给朋友时显示, 最多28个汉字"></el-input>
    </com-group>

    <!-- 上传功能实现
    1.没有上传过
    2.已经长传过显示上传的图片 -->
    <com-group title="微信分享卡片" tips="图片建议长宽为5:4" name-bock content-block
      >
      <el-button type="text" @click="shareImage = ''">重置</el-button>
      <up-load-box :img-url.sync="shareImage"></up-load-box>
    </com-group>

    <!-- 背景颜色 -->
    <com-group title="背景颜色">
      <el-button type="text" @click="backgroundColor = ''">重置</el-button>
      <!-- elui->颜色 -->
      <el-color-picker v-model="backgroundColor"></el-color-picker>
    </com-group>

    <!-- 背景图片 -->
    <com-group title="背景图片">
      <el-button type="text" @click="backgroundImage = ''">重置</el-button>
      <!-- 背景 -->
      <up-load-box :img-url.sync="backgroundImage"></up-load-box>
    </com-group>
  </div>
</template>

<script>
import ComTitle from '@/components/BasicUi/ComTitle'
import ComGroup from '@/components/BasicUi/ComGroup'
import ComDivider from '@/components/BasicUi/ComDivider'
import UpLoadBox from '@/components/BasicUi/UpLoadBox'

export default {
  name: 'SetPageInfo',
  components: {
    ComTitle,
    ComGroup,
    ComDivider,
    UpLoadBox
  },
  data() {
    return {
      // 初始背景颜色


    }
  },
  computed: {
    // computed中使用的是函数, 是直接从storestate中获取
    // 修改computed中定义的值,要写对象写法, get和set
    //修改对象下面的属性, 通过深拷贝去修改
    // 页面名称
    name: {
      get() {
        return this.$store.state.pageData.name
      },
      set(val) {
        this.upDatePageInfo({ name: val })
      },
    },
    // 微信分享文案
    shareDesc: {
      get() {
        return this.$store.state.pageData.shareDesc
      },
      set(val) {
        this.upDatePageInfo({ shareDesc: val })
      },
    },
      // 微信分享卡片
  shareImage: {
    get() {
        return this.$store.state.pageData.shareImage
    },
    set(val) {
      this.upDatePageInfo({ shareImage: val })
    },
  },
  // 背景颜色
  backgroundColor: {
    get() {
      return this.$store.state.pageData.backgroundColor
    },
    set(val) {
      this.upDatePageInfo({ backgroundColor: val })
    },
  },
  // 背景图片
  backgroundImage: {
    get() {
        return this.$store.state.pageData.backgroundImage
    },
    set(val) {
      this.upDatePageInfo({ backgroundImage: val })
    },
  },
  // 背景图片位置
  backgroundPosition: {
    get() {
        return this.$store.state.pageData.backgroundPosition
    },
    set(val) {
      this.upDatePageInfo({ backgroundPosition: val })
    },
  },

  },


  methods: {
    //更新配置数据
    upDatePageInfo(value) {
      this.$store.commit("SET_PAGE_CONFIG", value)
    }

  }
}
</script>
