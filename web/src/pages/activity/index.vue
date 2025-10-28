<template>
  <fc-table-search
    ref="table"
    :form-items="formItems"
    :table-columns="tableColumns"
    :form-config="{'label-width': '100px'}"
    @requestMethod="getTableData"
  >   
    <template #handleDetail="{row}">
      <el-button
        type="text"
        size="mini"
        @click="onEdit(row.id)"
      >编辑</el-button>
      <el-button
       type="text"
        size="mini"
        @click="onToggleActivity(row)"
      >{{row.isAbled === 0 ? '上线': '下线'}}</el-button>
      <el-button
        type="text"
        size="mini"
        @click="onPreview(row.id, row.title)"
      >预览</el-button>
      <el-button
        type="text"
        size="mini"
        @click="onDelete(row.id)"
      >删除</el-button>
    </template>
    <template slot="handleLeft">
      <el-button
        type="primary"
        @click="onAdd"
      >
        <i class="el-icon-plus"></i>
        新增页面
      </el-button>
    </template>
  </fc-table-search>
</template>

<script>
// import { checkBatchPageList } from '@/api/accountCheck/recordQuery'
// import { isEmpty } from '@/utils/index'
import {
  getCmsPageList,
  updateStatus,
  deletePage
} from '@/api/activity'
import { forEach } from 'lodash-es'
// import TableFlexHeight from '@/mixins/tableFlexHeight'
// import { activityStatus } from '@/config/activity'
export default {
  components: {},
  data () {
    return {
      formItems: [
        {
          comp: 'input',
          prop: 'name',
          label: "页面标题",
          clearable: true
        },
        {
          comp: 'select',
          prop: 'isAbled',
          label: '页面状态',
          options: [
            {
              value: 0,
              label: '下线'
            },
            {
              value: 1,
              label: '上线'
            }
          ],
          includeAll: false
        }
      ],
      tableColumns: [
        {
          prop: 'id',
          label: '页面ID'
        },
        {
          prop: 'name',
          label: '页面标题'
        }, 
        {
          prop: 'status',
          label: '页面状态'
        },
        {
          prop: 'create_time',
          label: '创建时间'
        },
        {
          prop: 'update_time',
          label: '更新时间'
        },
        {
          label: '操作',
          fixed: 'right',
          slotName: 'handleDetail'
        },
        
      ]
    }
  },
  methods: {
    async getTableData({current, size, ...tableData}, fn) {
      try {
        // console.log(tableData, 'tableData')
        let {...params} = tableData
        params.pageNum = current
        params.pageSize = size
        let res = await getCmsPageList(params)
        let {list, total} = res.data
        //后台定义的数据和使用的不同就用lodash-es里面的forEach
        forEach(list, item=> {
          //进行隐式转换1为true是上线 
          item.status = item.is_abled ? '上线': '下线'
          item.isAbled = item.is_abled
        })
        
        // console.log(res, 'res1')
        fn({
          data: list || [],
          total
        })
      }catch (err) {
        fn({message: err.message})
      }
    },
    onEdit(id) {
      this.$router.open({path: '/decorate', query: {id}})
    },
    onToggleActivity(row) {
      //上线
      if(row.isAbled === 0) {
        this.online(row, 1)
      } else {
        //下线
        this.offline(row, 0)
      }
    },
    online(row, isAbled) {
      //请求数据有loading效果
      this.$set(row, 'loading', true)
      updateStatus({id: row.id, isAbled}).then(()=> {
        this.$message.success('上线成功')
        //更新table页面
        this.$refs.table.onRefresh(true)
      }).catch(err=> {
        this.$message.warning(`操作失败${err.message}`)
      }).finally(()=> {
        //最后将loading设置为false
        this.$set(row, 'loading', false)
      })
    },
    offline(row, isAbled) {
      this.$confirm(`是否确认线下活动[${row.name}]?`, '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning',
        closeOnClickModel: false
      }).then(()=> {
        updateStatus({id: row.id, isAbled}).then(()=> {
          this.$message.success('下线成功')
          this.$refs.table.onRefresh(true)
        }).catch(err=> {
          this.$message.warning(`操作失败${err.message}`)
        })
      })
    },
    onPreview(id) {
      const url = this.$router.resolve({path: '/preview', query: {id, data: this.$store.state.wxParams}})
      window.open(url.href, '_blank')
    },
    onDelete(id) {
      this.$confirm(`是否确定删除?`, '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning',
        closeOnClickModel: false
      }).then(()=> {
        deletePage({id}).then(()=> {
          this.$message.success('删除成功')
          this.$refs.table.onRefresh(true)
        }).catch(err=> {
          this.$message.warning(`操作失败${err.message}`)
        })
      })
    },
    onAdd() {
      this.$router.open('/decorate')
    }
  }
}
</script>
