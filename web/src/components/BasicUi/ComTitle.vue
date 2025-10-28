
<template functional>
  <!-- 函数是组件: 组件内部没有逻辑,  仅仅依赖父组件传递过来的数据进行渲染,不用定义data, methods, 只有渲染效率高-->
  <div class="component-title">
    {{ props.title }}
  </div>
</template>

<style lang="less" scoped>
// 组件编辑标题
.component-title {
  margin-bottom: 14px;
  padding: 20px 16px;
  border-bottom: 1px solid #f2f4f6;
  font-size: 18px;
  font-weight: 600;
  line-height: 24px;
  color: #323233;
}
</style>
