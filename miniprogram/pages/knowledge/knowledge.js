// pages/knowledge/knowledge.js
const app = getApp()
const BASE_URL = app.globalData.apiBase
Page({

  /**
   * 页面的初始数据
   */
  data: {
  
    result:null,
    query:'',
    documents:[]
  },
  onLoad(){ this.loadDocuments() },
  onShow(){ this.loadDocuments() },
  onSearch(e){ this.setData({query:e.detail.value},()=>this.loadDocuments()) },
  loadDocuments(){
    wx.request({
      url:`${BASE_URL}/api/knowledge/documents`,
      data:{query:this.data.query},
      success:(res)=>{
        if(res.statusCode === 200) this.setData({documents:res.data.documents || []})
      }
    })
  },
  chooseFile(){
    wx.chooseMessageFile({
      count:1,
      type:'file',
      extension:['txt','doc','docx','pdf'],
      success:(res)=>{
        const file = res.tempFiles[0]
        wx.showLoading({title:'上传中...'})
        wx.uploadFile({
          url:`${BASE_URL}/api/knowledge/upload`,
          filePath:file.path,
          name:'file',
          success:(res)=>{
           try{
            const result=JSON.parse(res.data)
            //保存结果 
            if(!result.success) throw new Error('upload failed')
            this.setData({result:result})
            this.loadDocuments()
            wx.hideLoading()
            wx.showToast( {title:'上传成功',icon:'success'})
           }catch(e){
            this.setData({result:null})
            wx.hideLoading()
            wx.showToast({title:'上传失败',icon:'none'})
            
           }
          },
          fail:(res)=>{
            wx.hideLoading()
            wx.showToast({title:'无法与服务器通信',icon:'none'})
          }
        })
      }
      })
  }
})
