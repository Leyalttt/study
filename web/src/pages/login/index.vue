<template>
  <div class="login-bg login-container">
    <!-- 
      model	表单数据对象	object
      rules	表单验证规则	object
      label-position 表单域标签的位置，如果值为 left 或者 right 时，则需要设置
     -->
    <el-form 
      ref="loginForm" 
      :model="loginForm" 
      :rules="loginRules"
      class="login-form"
      label-position="left"
      >
      <div class="title-container">
        <h3 class="title">
          Cms内容管理系统
        </h3>
      </div>
      <el-form-item prop="username">
        <el-input 
          ref="username" 
          v-model="loginForm.username"
          placeholder="用户名"
          name="username"
          >
        </el-input>
      </el-form-item>
      <el-form-item prop="password">
        <el-input 
          ref="password"
          v-model="loginForm.password"
          placeholder="密码"
          name="password"
          type="password"
          >
        </el-input>
      </el-form-item>
      <el-button
        :loading="loading"
        type="primary"
        style="width: 100%; margin-bottom: 30px"
        @click="handleLogin"
      >登录</el-button>
    </el-form>
  </div>
</template>

<script>
import { login } from '@/api/activity'
export default {
  name: 'Login',
  data() {
    const validateUsername = (rule, value, callback)=> {
      if(!value) {
        callback(new Error("请输入用户名"))
      } else {
        callback()
      }
    }
    const validatePassword = (rule, value, callback)=> {
      if(value.length < 6) {
        callback(new Error("请输入密码"))
      } else {
        callback()
      }
    }
    return {
      loginForm: {
        username:'',
        password: ''
      },     
      loginRules: {
        username: [{ required: true, trigger: 'blur', validator: validateUsername }],
        password: [{ required: true, trigger: 'blur', validator: validatePassword }]
      },
      loading: false
    }
  },
  methods: {
    handleLogin() {
      this.$refs.loginForm.validate(valid => {
        if(valid) {
          this.loding = true
          login(this.loginForm).then(res=> {
            this.loading = false
            localStorage.setItem("token", res.data.token)
            localStorage.setItem("username", res.data.username)
            this.$router.replace("/home")
          }).catch(err=> {
            this.loading = false
          })
        } else {
          return false
        }
      })
    }

  }

}
</script>
<style lang="less" scoped>
.login-bg {
  // width: 100%;
  min-width: 1200px;
  height: 100%;
  background-image: url('../../assets/images/SAAS_login/bg_login.jpg');
  background-size: 100% 100%;
  background-repeat: no-repeat;
}
.login-container {
  .el-input {
    display: inline-block;
    height: 100%;
    width: 100%;
    input {
      background: transparent;
      border: 0px;
      -webkit-appearance: none;
      border-radius: 0px;
      padding: 12px 5px 12px 15px;
      color: #fff;
      height: 47px;
      caret-color: #fff;
    }
  }
}
.login-container {
  position: relative;
  min-height: 100%;
  width: 100%;
  overflow: hidden;

  .login-form {
    position: absolute;
    right: 180px;
    width: 400px;
    max-width: 100%;
    padding: 160px 35px 0;
    overflow: hidden;
  }

  .tips {
    font-size: 14px;
    color: #fff;
    margin-bottom: 10px;

    span {
      &:first-of-type {
        margin-right: 16px;
      }
    }
  }

  .svg-container {
    padding: 6px 5px 6px 15px;
    color: #889aa4;
    vertical-align: middle;
    width: 30px;
    display: inline-block;
  }

  .title-container {
    position: relative;

    .title {
      font-size: 26px;
      color: rgb(29, 24, 24);
      margin: 0px auto 40px auto;
      text-align: center;
      font-weight: bold;
    }
  }

  .show-pwd {
    position: absolute;
    right: 10px;
    top: 7px;
    font-size: 16px;
    color: #889aa4;
    cursor: pointer;
    user-select: none;
  }

  .thirdparty-button {
    position: absolute;
    right: 0;
    bottom: 6px;
  }
}
</style>
