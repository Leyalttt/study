import Vue from 'vue'
import Router from 'vue-router'
import Layout from '@/layout'
import './routerExtend'

export const constantRoutes = [
  {
    path: '/',
    component: Layout,
    redirect: '/login',
    children: [
      {
        path: '/home',
        component: () => import('@/pages/home/index.vue'),
        name: '首页',
        hidden: true,
        meta: { title: '首页' }
      }
    //   {
    //     path: "/editPassword",
    //     component: () => import("../pages/editPassword/index.vue"),
    //     name: "editPassword",
    //     hidden: true,
    //     meta: { title: "修改密码" },
    //   },
    ]
  },
  {
    path: "/login",
    component: () => import("../pages/login/index.vue"),
    name: "Login",
    hidden: true,
    meta: { title: "登录" },
  },
  {
    path: '/404',
    component: () => import('../pages/error-page/404'),
    name: '404',
    hidden: true,
    meta: { title: '404' }
  },
  {
    path: '/401',
    component: () => import('../pages/error-page/401'),
    name: '401',
    hidden: true,
    meta: { title: '401' }
  },
  // 搭建页面
  {
    path: '/decorate',
    name: 'decorate',
    component: () => import('../pages/decorate/index')
  },
  // 预览页面
  {
    path: '/preview',
    name: 'preview',
    component: () => import('../pages/preview/index')
  },
  // 活动列表页
  {
    path: '/activity',
    name: 'activity',
    component: Layout,
    meta: { title: '活动管理' },
    children: [
      {
        path: '/',
        component: () => import('@/pages/activity/index'),
        hidden: true
      }
    ]
  }
]
// 异步挂载的路由
// 动态需要根据权限加载的路由表
export const asyncRoutes = []

const createRouter = () =>
  new Router({
    scrollBehavior: () => ({ y: 0 }),
    routes: constantRoutes
  })
const router = createRouter()
Vue.router = router
// Detail see: https://github.com/vuejs/vue-router/issues/1234#issuecomment-357941465
export function resetRouter() {
  const newRouter = createRouter()
  router.matcher = newRouter.matcher // reset router
}

export default router
