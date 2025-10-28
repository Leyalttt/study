import Vue from 'vue'
import App from '@/App.vue'
import store from '@/store'
import router from '@/router'
import VueRouter from 'vue-router'
import '@/styles/app.less' // 后台装修搭建系统公共CSS
import VueQuillEditor from 'vue-quill-editor'
import 'quill/dist/quill.core.css'
import 'quill/dist/quill.snow.css'
import 'quill/dist/quill.bubble.css'

import PizzaUI from '@bigbighu/mobili-pizza'
import CoffeeUI from '@bigbighu/mobili-coffee'

// import ElementUI from 'element-ui';
// import 'element-ui/lib/theme-chalk/index.css';
Vue.config.productionTip = false
Vue.use(VueRouter)
Vue.use(VueQuillEditor)
Vue.use(PizzaUI)
Vue.use(CoffeeUI)
// Vue.use(ElementUI);
new Vue({
  el: '#app',
  router,
  store,
  render: h => h(App)
})
