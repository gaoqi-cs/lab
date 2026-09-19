// pages/home/home.js
//获取app实例
const app = getApp()

Page({

  /**
   * 页面的初始数据
   */
  data: {
    latestInspection:null,
    features:[
      {title:'拍照隐患识别',description:'AI 快速识别潜在风险',icon:'/images/classify_active.png',url:'/pages/classify/classify',tone:'blue'},
      {title:'场景模型训练',description:'定制本校安全类别模型',icon:'/images/me_active.png',url:'/pages/train/train',tone:'orange'},
      {title:'安全规程库',description:'随时查询，规范操作',icon:'/images/history_active.png',url:'/pages/knowledge/knowledge',tone:'blue'},
      {title:'智能安全问答',description:'有问必答，安全顾问',icon:'/images/chat_active.png',url:'/pages/chat/chat',tone:'orange'}

    ]
  },
  onShow(){
    wx.request({
      url:`${app.globalData.apiBase}/api/inspections/`,
      data:{limit:1},
      success:(res)=>{
        if(res.statusCode === 200) this.setData({latestInspection:(res.data.records || [])[0] || null})
      }
    })
  },
  navigateTo(e){
    // 获取点击的url
    const url = e.currentTarget.dataset.url
    if(!url) return
    const tabBarPages = [
      '/pages/home/home',
      '/pages/train/train',
      '/pages/classify/classify',
      '/pages/knowledge/knowledge',
      '/pages/chat/chat'
    ]

    if(tabBarPages.includes(url)){
      //如果点击的是页面
      wx.switchTab({
        url,
        fail: (error) => {
          console.error('切换页面失败', url, error)
          wx.showToast({title: '页面打开失败，请重新编译', icon: 'none'})
        }
      })
    }
    else{
      //点击导航栏
      wx.navigateTo({
        url,
        fail: (error) => {
          console.error('打开页面失败', url, error)
          wx.showToast({title: '页面打开失败，请重新编译', icon: 'none'})
        }
      })
    }
  }
})
