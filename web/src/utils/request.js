//引入axios
import axios from'axios'
import {baseURL} from'../config/index'
import {Message} from'element-ui'

//创建axios实例
const service = axios.create({
 // baseURL是公共部分,在config有定义引入 
 baseURL,
 timeout: 2000,
}) 

//添加请求拦截器
service.interceptors.request.use(config=>{
  //请求token信息
  const token = localStorage.getItem('token')
  //如果token存在就给请求头加上token
  if(token) {
    config.headers['X-token'] = token
  }
  return config
},error=> {
  return Promise.reject(error)
})

//添加响应拦截器
service.interceptors.response.use(response=> {
  const res = response.data
  //不等于10000就是请求错误
  if(res.code !== 10000) {
    Message.err(res.message)
  }
  //登录超时
  if (res.code === -2) {
    //清空token, 跳转至登录页
    localStorage.removeItem('token')
    window.location.href = location.origin + '/cms-manage/#/login'
  }
  //正常返回res
  return res
}, error=> {
  return Promise.reject(error)
})

export default service