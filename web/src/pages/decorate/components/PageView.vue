<!-- 中间内容区域 -->
<template>
  <div ref="pageView" class="page-view">
    <!-- 手机预览模块 -->
    <div class="preview">
      <div class="preview-head">
        <div class="preview-head-title">
          {{pageData.name || '微页面标题'}}
        </div>
      </div>
      <div class="preview-wrap">
        <!-- 通过id来获取iframe 
              @load="onloadH5" iframe加载完会触发这个函数
        -->
        <iframe 
          id="previewIframe"
          class="preview-iframe"
          :src="previewSrc"
          title="频道名称"
          frameborder="0"
          allowfullscreen
          width="100%"
          :height="previewHeight"
          @load="onloadH5"
            >
        </iframe>
      </div>

    <!-- 监听拖拽进入 -->
    <!-- @dragover 拖进一个有效的放置目标时（每几百毫秒）触发。 -->
    <div v-if='dragActive' class="preview-drag-mask" @dragover="onDragover"></div>
    <div v-if='dragActive' class="preview-drag-out" @dragover="onDragout"></div>
    <!-- 监听拖拽离开 -->
    </div> 
  </div>
</template>

<script>
import { mapState, mapActions, mapMutations } from 'vuex'
import settings from '@/config'
export default {
  name: 'PageView',
  data() {
    return {
 
    }
  },
  computed: {
   ...mapState(['pageData', 'previewHeight', 'dragActive', 'componentsTopList', 'addComponentIndex']),
   
  //  路径拼接, :src 路径 存放在config-> index.js中
  // &noLogin=true 是否需要登录
   previewSrc() {
    return settings.decorateViewSrc + `?pageId=${this.$route.query.id || ''}&noLogin=true"`
   }
  },
  mounted() {
    // console.log(this.dragActive, 'dragActive')
    // console.log(this.pageData, 'preview')
    //完成跨源通信实例的监听
    this.initMessage()
  },
  methods: {
    ...mapActions(["initMessage", "pageChange"]),
    ...mapMutations(['SET_DRAG_INDEX', 'VIEW_ADD_PREVIEW', 'VIEW_DELETE_PREVIEW']),
    // 左侧组件拖动到页面预览区域事件
    onDragover(event) {
      // console.log('触发onDragover')
      //禁用拖拽的默认事件
      event.preventDefault()
      //超出浏览器: 浏览器顶部到鼠标的高度+ 浏览器滚动高度 - 空白高度
      let dropTop = event.pageY + this.$refs.pageView.scrollTop - 191
      //获取当前拖动组件要添加的位置索引
      let addIndex = 0
      // console.log(this.componentsTopList, 'componentsTopList')
      //遍历所有组件的高度, 倒序
      for(let i = this.componentsTopList.length - 1; i >= 0 ;i--) {
        //元素的高度
        const value = this.componentsTopList[i]
        //上一个元素的高度
        const prev = this.componentsTopList[i - 1] || 0
        //当前元素的中间高度 = (元素的高度-上一个元素的高度) / 2
        const _half = (value - prev) / 2
        //如果i===0 并且鼠标的高度小于一半, 那么拖拽的组件是放在了第一位, 所以直接break
        if(i=== 0 && dropTop <= _half) break
        if(dropTop > (value - _half)) {
          addIndex  = i + 1
          break
        }
      }
      //判断是否需要进行跨源通信
      if (addIndex === this.addComponentIndex) return 
      //将与添加的索引放到store
      //跨源通信传递数据给crs
      this.SET_DRAG_INDEX(addIndex)
      // 向H5页面发送预添加组件
      this.VIEW_ADD_PREVIEW(addIndex)
    },

    onDragout(event) {
      console.log('触发onDragout')
      event.preventDefault()
      //如果组件与添加了//移除欲添加组件
      if (this.addComponentIndex !== null) {
        if (this.addComponentIndex != null) {
        // console.log('预删除组件')
        this.SET_DRAG_INDEX(null)
        this.VIEW_DELETE_PREVIEW()
      }
      }
    },
    onloadH5() {
      console.log('iframe加载完成')
      //跨源通信
       this.$store.commit("VIEW_UPDATE")
    }

  }

}
</script>

<style lang="less" scoped>

.page-view {
  position: absolute;
  top: 56px;
  left: 186px;
  right: 376px;
  bottom: 0;
  overflow-y: auto;
  overflow-x: hidden;
  background-color: #f7f8fa;
  display: flex;
  justify-content: center;
  user-select: none;
}
.cache-box {
  position: absolute;
  left: 20px;
  right: 20px;
  top: 20px;
  height: 30px;
  line-height: 30px;
  overflow: hidden;
  background: @color-1-bg;
  border: 1px solid @color-1;
  padding: 0 10px;
  z-index: 2;
  .color-1 {
    cursor: pointer;
  }
  .cache-close {
    position: absolute;
    right: 0;
    top: 0;
    width: 30px;
    height: 30px;
    text-align: center;
    cursor: pointer;
    color: #aaa;
  }
}
.preview {
  position: absolute;
  width: 100%;
  .preview-head {
    height: @content-header-height;
    width: 375px;
    margin: 72px auto 0;
    background: url("~@/assets/img/layout/header_bg.png") left top no-repeat;
    background-size: cover;
    position: relative;
    box-shadow: 0 0 14px 0 rgba(0, 0, 0, 0.1);
    .preview-head-title {
      width: 180px;
      margin: 0 auto;
      height: 64px;
      font-size: 14px;
      text-align: center;
      padding-top: 30px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
  }
  .preview-iframe {
    min-height: 600px;
  }
  .preview-drag-mask {
    position: absolute;
    left: 50%;
    top: 60px;
    bottom: 20px;
    width: 520px;
    margin-left: -260px;
    z-index: 10;
  }
  .preview-drag-out {
    position: absolute;
    left: 0;
    right: 0;
    top: 0;
    bottom: 0;
    z-index: 9;
  }
}

// .preview-drag-mask {
//   background-color: aqua;
//   position: absolute;
//   left: 50%;
//   top: 60px;
//   bottom: 20px;
//   width: 520px;
//   margin-left: -360px;
//   z-index: 10;
// }
// .preview-drag-out {
//   background-color: blueviolet;
//   position: absolute;
//   left: 0;
//   top: 0;
//   bottom: 0;
//   width: 0;
// }
</style>
