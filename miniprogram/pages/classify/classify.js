// pages/classify/classify.js
const USER_DATA_PATH = wx.env.USER_DATA_PATH
const fs = wx.getFileSystemManager()
const app = getApp()
Page({
  data: {
    result:null,
    resultStr:'',
    isProcessing:false,
    image:{path:'',name:'',size:0,url:''},
    error:null,
    isSaved:false
  },
  takePhoto(){
    wx.chooseMedia({
      count: 1,
      mediaType:['image'],
      sourceType:['camera'],
      camera:'back',
      sizeType:['original'],
      success:(res)=>{
        const tempFile = res.tempFiles[0]
        this.copyAndAddImage(tempFile.tempFilePath,tempFile.size)
      }
  })
},
  chooseFromAlbum(){
    wx.chooseMedia({
      count: 1,
      mediaType:['image'],
      sourceType:['album'],
      sizeType:['compressed'],
      success:(res)=>{
        const tempFile = res.tempFiles[0]
        this.copyAndAddImage(tempFile.tempFilePath,tempFile.size)
      }
    })
  },
  saveTemporaryResult(){
    if (!this.data.result || this.data.isSaved) return
    const inspection = this.data.result.result || {}
    wx.request({
      url:`${app.globalData.apiBase}/api/inspections/`,
      method:'POST',
      data:{image_name:this.data.image.name || '现场图片',label:String(inspection.top1 || '待确认'),confidence:String(inspection.top1_confidence || '未知')},
      success:(res)=>{
        if(res.statusCode === 201){
          this.setData({isSaved:true})
          wx.showToast({title:'巡检记录已保存',icon:'success'})
        }else wx.showToast({title:'保存失败，请重试',icon:'none'})
      },
      fail:()=>wx.showToast({title:'无法连接后端服务',icon:'none'})
    })
  },
  copyAndAddImage(tempPath,size)
  {
    const fileName = `img_${Date.now()}_${Math.random().toString(36).substr(2,6)}.jpg`
    const cachePath = `${USER_DATA_PATH}/${fileName}`
    try{
      //将临时文件复制到用户目录下
      fs.copyFileSync(tempPath,cachePath)
      const name = tempPath.split('/').pop() || 'image.jpg'
      this.setData({image:{path:cachePath,name,size,url:cachePath},result:null,error:null,isSaved:false})
      console.log('复制图片成功')

    }catch(error){
      console.error('复制图片失败:',error)
      wx.showToast({title:'复制图片失败',icon:'none'})
    }
    //this.startRecognize()

  },
  

  startRecognize() {
    if (!this.data.image.path || this.data.isProcessing) return
    this.setData({isProcessing:true,error:null})
    fs.readFile({
      filePath:this.data.image.path,
      success:(res)=>{
        const base64 = wx.arrayBufferToBase64(res.data)
        const ext =this.data.image.name.split('.').pop() || 'jpg'
        const mimeType = `image/${ext === 'jpg' ? 'jpeg' : ext}`
        const param_data = `data:${mimeType};base64,${base64}`
        wx.request({
          url: `${app.globalData.apiBase}/api/classify`,
          method: 'POST',
          data: {
            image: param_data
          },
          header:{
            'Content-Type':'application/json'
          },
          timeout:(60*1000),
          success: (res) => {
           if(res.statusCode === 200){
           this.setData({result:res.data,
              resultStr: JSON.stringify(res.data),
              error:null,
              isProcessing:false,
              isSaved:false})
           }
           else {
            this.setData({error:'服务器未返回有效判别结果',isProcessing:false})
           }
           console.log('识别结果',res)
          },
          fail:(err)=>{
        this.setData({error:'判别失败，请检查后端服务和模型资源',isProcessing:false})
            console.error('识别失败',err)
            wx.showToast({title:'识别失败',icon:'none'})
          }
        })
      },
      fail:()=>this.setData({error:'读取现场图片失败',isProcessing:false})
    })



  },
  




  /**
   * 生命周期函数--监听页面显示
   */
  onShow() {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide() {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload() {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh() {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom() {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage() {

  }
})
