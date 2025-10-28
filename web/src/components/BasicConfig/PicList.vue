<template>
  <div>
    <!-- 拖拽容器
    个数大于1才会拖动 -->
    <draggable v-model="picData" :options="{}">
    <div
    v-for="(item, index) in picData" 
      :key="index" 
      class="up-pic-item" 
      :class="{'can-drag': picData.length > 1}"
    >
      <div class="up-pic-item-wrapper">
            <up-load-box
              v-if="showPic"
              :img-url.sync="item.imageUrl"
            ></up-load-box>
              <div class="info">
                <dl class="com-form-group">
                  <dt class="form-label">标题</dt>
                  <dd class="form-container">
                    <el-input v-model.lazy="item.text" :placeholder="addPlaceHolder"></el-input>
                  </dd>
                </dl>
                <dl class="com-form-group">
                  <dt class="form-label">链接</dt>
                  <dd class="form-container">
                    <!-- 弹窗组件 -->
                    <config-link :link-obj.sync="item.link"></config-link>
                  </dd>
                </dl>
              </div>    
          </div>
          <!-- 添加图片 -->
          <ButtonAdd v-if="showAdd" :add-text="addPlaceHolder" @click="addItem" />
      </div>
    </draggable>
  </div>
</template>

<script>
import draggable from 'vuedraggable' // 拖拽元素
import UpLoadBox from '@/components/BasicUi/UpLoadBox'

import ButtonAdd from '@/components/BasicUi/ButtonAdd'
import ConfigLink from '@/components/BasicConfig/ConfigLink'

export default {
  name: 'PicList',
  components: {
    draggable,
    UpLoadBox,
    ButtonAdd,
    ConfigLink
  },
  props: {
    // 图片列表数组
    imageList: {
      type: Array, default: null
    },
    // 图片标题文本输入框缺省提示文字
    inputPlaceHolder: {
      type: String, default: ''
    },
    // 添加图片文字
    addPlaceHolder: {
      type: String,
      default: '添加广告图'
    },
    // 是否显示图片选项
    showPic: {
      type: Boolean,
      default: true
    },
    // 是否显示标题选项
    showName: {
      type: Boolean,
      default: true
    },
    // 是否显示添加按钮
    showAdd: {
      type: Boolean,
      default: true
    },
    // 是否显示删除按钮
    showDelete: {
      type: Boolean,
      default: true
    },
    // 列表是否不可拖拽排序
    unGraggable: {
      type: Boolean,
      default: false
    },
    // 最多添加的数据条目数
    limitSize: {
      type: Number,
      default: 10
    }
  },
  data() {
    return {
      //props传递过来的数据不能直接修改, 要进行深拷贝
      picData: JSON.parse(JSON.stringify(this.imageList)),
    }
  },
  watch: {
    //监听当前图片列表数据修改
    picData: {
      handler: function(newVal) {
        // console.log(newVal, 'newVal')
        this.$emit('update:imageList', newVal)
      },
      deep: true
    }
  },
  methods: {
    // 删除链接
    deleteItem(index) {
      this.picData.splice(index, 1)
    },
    //添加广告图
    addItem() {
      //有图片的
      if (this.showPic) {
        this.showDialogImage()
      } else {
        //不带图片文本类型
        this.picData.push({
          link: null,
          imageUrl: '',
          text: ''
        })
      }
      
    },
    //图片上传
    showDialogImage() {
      //将弹窗属性设为true
      this.$store.commit('SET_UPIMAGE_VISIBLE', true)
      //修改上传图片成功的回调事件SET_UPIMAGE_FUC
      this.$store.commit('SET_UPIMAGE_FUC', this.uploadImgSuccess)
    },
    //图片上传成功
    uploadImgSuccess(imgUrl) {
      this.picData.push({
        link: null,
        imageUrl: imgUrl,
        text: ''
      }) 
    }
  }
}
</script>

<style lang="less" scoped>
// 图片列表
.up-pic-item {
  position: relative;
  margin-bottom: 12px;
  padding: 6px 0;
  background: #ffffff;
  box-shadow: 0 0 4px 0 rgba(10, 42, 97, 0.2);
  border-radius: 2px;
  user-select: none;
  &.can-drag {
    cursor: move;
  }
  &.sortable-ghost {
    opacity: 0.2;
  }
  &:hover {
    .up-pic-item-delete {
      visibility: visible;
    }
  }
  /deep/.config-link .cllt-name{
    max-width:110px
  }
}
.up-pic-item-wrapper {
  display: flex;
  .up-pic-box {
    margin: 6px 0 0 12px;
  }
  .info {
    flex: 1;
  }
}
// 图片编辑表单
.com-form-group {
  padding: 6px 12px;
  // display: flex;
  align-items: center;
  .form-label {
    margin-right: 16px;
    font-size: 14px;
    color: #323233;
    line-height: 18px;
    white-space: nowrap;
  }
  .form-container {
    flex: 1;
  }
  /deep/ .input-name .el-input__inner {
    height: 32px;
    line-height: 32px;
    padding: 0 10px;
    border-radius: 2px;
  }
}
</style>
