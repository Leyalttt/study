<template>
  <div class="page-head">
    <a class="go-back" href="javascript:;" @click="backToList">
      <img class="logo" src="@/assets/img/layout/logo.png" alt="">
      返回频道列表页面
    </a>
    <div class="page-operate">
      <!-- loading	是否加载中状态	boolean 默认为false -->
      <el-button 
        size="medium" 
        type="primary" 
        @click="saveAndCountinue"
        :loading="saveLoading"
        >保存
      </el-button>
      <el-button 
        size="medium" 
        @click="saveAndView"
        :loading="previewLoading"
        >预览
      </el-button>
    </div>
  </div>
</template>

<script>

import {saveCmsPage} from'@/api/activity'
export default {
  name: 'PageHead',
  data() {
    return {
      saveLoading: false,
      previewLoading: false
    }
  },
  computed: {
    pageData() {
      return this.$store.state.pageData
    }
  },
  methods:{
    //返回至列表页
    backToList() {
      //为了处理父窗口是活动列表的情况, 关闭父窗口, 在当前窗口跳转至活动列表
      try {
        //父窗口有值并且父窗口的地址是#/activity
        if(window.openr && window.openr.location.hash === '#/activity') {
        //关闭父窗口
          window.opener.close()
        }
        //路由跳转回activity
        this.$router.push('/activity')
      } catch (error) {
        this.$router.push('/activity')
      }
      //要判断当前详情页是不是从详情页跳转过来的
      //window.openr获取父级的窗口
      //如果是新开的地址, window.openr是null
      //window.openr.location.hash 可以拿到上一层的url地址
      
    },
    //保存
    saveAndCountinue() {
      this.saveLoadng = true
      this.savePage().then(()=> {
        this.$message.success('保存成功')
        console.log(this.$message, '保存成功')
      }).catch((err)=> {
        this.$message.warning(`保存失败: ${err.message}`)
      }).finally(()=> {
        this.saveLoadng = false
      })
    },

    //预览
    saveAndView() {
      this.previewLoading = true
      this.saveCmsPage({online: 1}).then((res)=> {
        this.$message.success('保存成功')
        const hasId = res && res.data? res.data.id: ''
        //详情页分为新增和编辑, 编辑携带id, 新增从后台里面拿到id
        if (hasId) {
          this.goToView(hasId)
        } else {
          this.goToView(this.$route.query.id)
        }
      })
    },
    //保存和预览都有共性的逻辑, 都要去保存页面数据
    savePage(params) { 
      //参数合并
     const pageData =  {...this.pageData, ...params}
     // 保存时,给每个组件加上sort用于标记id
      pageData.componentList.forEach((item, index)=> {
        item.sort = index
      })
      return saveCmsPage(pageData).then(res=> {
        // console.log(res, 'res')
        //判断如果是新增, 需要调用接口后再pageData中添加id
        if(res && res.data && res.data.id) {
          const cloneData = JSON.parse(JSON.stringify(this.pageData))
          cloneData.id = res.data.id
          this.$store.commit('UPDATE_COMPONENT', cloneData)
          this.$router.push(`/decorate?id=${res.data.id}`)
        }
      })
    },
    //跳转预览页面
    goToView(id) {
      const url = this.$router.resolve({
        path: 'preview',
        query: {id}
      })
      window.open(url.href, '_blank')
    }

  },
}
</script>

<style lang="less" scoped>
  .page-head {
    position: fixed;
    left: 0;
    top: 0;
    width: 100%;
    height: @page-header-height;
    background: #fff;
    border-bottom: 1px solid #ebedf0;
    padding: 0px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    .go-back {
      float: left;
      line-height: 55px;
      font-size: 14px;
      color: #4f4f4f;
      cursor: pointer;
      .logo {
        display: inline-block;
        vertical-align: middle;
        margin: -2px 4px 0 0;
        max-width: 30px;
        max-height: 30px;
      }
    }
  }
</style>
