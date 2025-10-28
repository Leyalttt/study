<template>
  <div class="home">
    <!-- 搭建框架 -->
    <!-- 顶部header -->
    <PageHead />
    <!-- 左侧组件列表 -->
    <PageLeft />
    <!-- 中间内容区域 -->
    <PageView />
    <!-- 右侧组件编辑区域 -->
    <PageRight />

    <!-- 组件的公共配置内容 -->
    <!-- 上传图片组件 -->
    <UpLoadImg :dialog-image-visible.sync="dialogImageVisible" @upLoadImgSuccess="upLoadImgSuccess" />
  </div>
</template>
<script>
// 结构组件
import PageHead from './components/PageHead'
import PageLeft from './components/PageLeft'
import PageView from './components/PageView'
import PageRight from './components/PageRight'
//生成随机数字
import { createRandomId } from '@/utils'
// 公共配置组件
import UpLoadImg from '@/components/BasicConfig/UpLoadImg'
// 获取页面数据
import { getCmsPageById } from '@/api/activity'
export default {
  name: 'App',
  components: {
    PageHead,
    PageLeft,
    PageView,
    PageRight,
    UpLoadImg
  },
  // data() {
  //   return {
  //   }
  // },
  created() {
    this.init()
  },
  methods: {
    init() {
      this.getData()
    },
    //获取详情页数据进行封装, 因为多个地方使用
    getData() {
      //接口接受id, 从点击列表页跳转到详情页, id在列表页里面拿到数据携带在url上传递给详情页
      //通过当前路由拿到id
      let id = this.$route.query.id
      // console.log(id, 'id')
      //点击编辑会跳转到详情页id存在, 点击增加也会跳转到详情页id不存在
      if (id) {
        getCmsPageById(id).then(({data})=> {
          console.log(data, 'data')
          this.postDataToH5(data)
        })
      }
      //
    },
    //处理后端返回的数据, 将数据存入到store中
    postDataToH5(data) {
      //要进行容错, data 和 data.component存在
      if(data && data.componentList) {
        data.componentList.forEach(item => {
          //如果时间是string类型要进行转换
          if (item.data.validTime && typeof item.data.validTime === 'string') {
            item.data.validTime = JSON.parse(item.data.validTime)
          }
          //新增的页面没有id要增加id, 编辑的页面人家自带id
          if(!item.id) {
            //随机生成数字
            const id = createRandomId()
            item.id = item.data.component + '-' + id
          }
        });
        //更新页面
        this.$store.commit('UPDATE_COMPONENT', {data})
        this.$store.commit('VIEW_UPDATE')

      }
      
    },
    upLoadImgSuccess(imgUrl) {
      if(this.state.uploadImgSuccess) {
        this.state.uploadImgSuccess(imgUrl)  
      }
    }

  },
  computed: {
    dialogImageVisible: {
      get() {
        return this.$store.state.dialogImageVisible
      },
      set(val) {
        this.$store.commit("SET_UPIMAGE_VISIBLE", val)
      }
    }
  }
}
</script>
