// pages/chat/chat.js
const app = getApp()
const BASE_URL = app.globalData.apiBase
Page({

  /**
   * 页面的初始数据
   */
  data: {
    messages:[{role:'ai',content:'我是安全操作助手。请描述你正在进行的实验步骤，或直接询问已导入规程中的要求。',time:''}],
    inputText:'',
    isStreaming:false,
    scrollIntoView:''

  },
  onInput(e){
    //输入框内容改变时，更新inputText
    this.setData({inputText:e.detail.value})
  },
  useSuggestion(e){
    this.setData({inputText:e.currentTarget.dataset.text})
  },
getTime(){
  //将js的时间戳转换为自定格式的字符串
  const now = new Date()
  return `${now.getHours().toString().padStart(2,'0')}:${now.getMinutes().toString().padStart(2,'0')}`
},
  scrollBottom(){
    this.setData({scrollIntoView:'bottom'})
  },
  sendMessage(){
    //获得页面的输入(用户的问题)
    const text = this.data.inputText.trim()
    //如果输入为空，或者正处于问答状态，则不发送
    if(!text || this.data.isStreaming) return
    const userMsg = {
      role:'user',
      content:text,
      time:this.getTime()
    }
    // 保存用户的问题到messages数组中，同时将isStreaming设置为true
    this.setData({messages:[...this.data.messages,userMsg]
    ,inputText:''
    ,isStreaming:true
    
  })
  //滚动到底部
  this.scrollBottom()
  //ai 回答
  const aiMsg = {
    role:'ai',
    content:'',
    time:this.getTime()
  }
  //保存ai回答到messages数组中,防止用户等待最终的回答时间过长
  //同时确保及时超过token限制,也能显示前半部分的答案
  this.setData({messages:[...this.data.messages,aiMsg]})
  let timeoutId = this.checkChunkTimeout()
    const requestTask = wx.request({
      url:`${BASE_URL}/api/chat/stream`,
      method:'POST',
      data:{
        message:text
        ,temperature:0.7
        ,max_tokens:512
        
      },
      header:{
        'Content-Type':'application/json'
      },
      //流式交互,需要设置chunked编码,还需要设置responseType为arraybuffer(缓冲区)
      enableChunked:true,
      responseType:'arraybuffer',
      success:(res)=>{
        //因为流式交互会多次交互所以不能在这里交互成功的处理逻辑
      },
      fail:(err)=>{
        //清空计时器
        clearTimeout(timeoutId)
        //交互状态清空
        this.setData({isStreaming:false})
        wx.showToast({title:'交互失败',icon:'none'})
        console.error('交互失败',err)
      }
    })
    //定义缓冲区变量
    let buffer = ''
    //流式交互每次接收到的数据是一个chunk，需要拼接起来，然后解析
    requestTask.onChunkReceived((res)=>{
      const uint8Array = new Uint8Array(res.data)
      const str = this.utf8ArrayToString(uint8Array)
      //拼凑来自后端的答案（片段）
      buffer +=str
      const lines = buffer.split('\n')
      //如果最后一行没有形成完整的行(\n结尾)，保存该行，拼接下一次收到的信息，但如果最后一行是完整的，则直接解析该行并清空缓存
      buffer = lines.pop() || ''
      for (const line of lines){
        if (line.startsWith('data:')){
          //真正的信息在data:之后
          const jsonStr =line.substring(5).trim()
          if (!jsonStr) continue
          try{
            const data = JSON.parse(jsonStr)
            if (data.done){
              //后端发送完毕
              this.onDone(timeoutId)
            }else if(data.chunk !== undefined){
              //后续还有信息
              timeoutId = this.onChunk(data.chunk,timeoutId)
            }else if(data.error){
              //如果发生错误
              this.onError(data.error,timeoutId)
            }
          }catch(e){
            console.error('解析JSON失败:',e)
          }

        }
      }

    })

  },
  utf8ArrayToString(uint8Array){
    let out = ''
    let i = 0
    let len = uint8Array.length
    let c,char2,char3
    while(i < len){
      c = uint8Array[i++]
      switch(c >> 4){
        case 0: case 1: case 2: case 3: case 4: case 5: case 6: case 7:
          out +=String.fromCharCode(c)
          break
        case 12:
        case 13:
          char2 = uint8Array[i++]
          out +=String.fromCharCode((c & 0x1F) << 6 | (char2 & 0x3F))
          break
        case 14:
          char2 = uint8Array[i++]
          char3 = uint8Array[i++]
          out +=String.fromCharCode((c & 0x0F) << 12 | (char2 & 0x3F) << 6 | (char3 & 0x3F))
          break
        default:
          break
    }
  }
  return out
},
  onDone(){
    //后端发送完毕,停止交互
    //清空计数器
    if(this.chunkTimerId)
    clearTimeout(this.chunkTimerId)
    
    //交互状态清空
    this.setData({isStreaming:false})
    wx.showToast({title:'交互完成',icon:'none'})
  },
  onChunk(chunk, timeoutId){
    //每次片段交互
    timeoutId = this.checkChunkTimeout(timeoutId)
    //为下一次片段计时
    const msgs = this.data.messages
    const last = msgs[msgs.length - 1]
    if(last.role === 'ai'){
      // ai消息，拼接内容
      last.content += chunk
      this.setData({messages:msgs})
      //滚动到底部
      this.scrollBottom()
    }
    return timeoutId
    //为下一次片段计时
    
    

  },
  onError(err,timeoutId){
    //交互失败
    if(timeoutId)
     clearTimeout(timeoutId)
     this.setData({isStreaming:false})
     wx.showToast({title:'交互失败',icon:'none'})
     console.error('交互失败',err)
     const msgs = this.data.messages
     if (msgs.length && msgs[msgs.length - 1].role === 'ai' && msgs[msgs.length - 1].content === ''){
       //删除最后一个ai消息,因为ai消息是空的
       msgs.pop()
       this.setData({messages:msgs})
     }
    
  },
  checkChunkTimeout(timeoutId){
    if (timeoutId)
     clearTimeout(timeoutId)
    return setTimeout(()=>{
      if (this.data.isStreaming){
        //思考超时停止交互
          this.setData({isStreaming:false})
          wx.showToast({title:'思考超时',icon:'none',duration:3000})
          //如果第一次接受的消息就超时
          const msgs = this.data.messages
          if (msgs.length && msgs[msgs.length - 1].role === 'ai' && msgs[msgs.length - 1].content === ''){
            //删除最后一个ai消息,因为ai消息是空的
            msgs.pop()
            this.setData({messages:msgs})
          }
       }
    },180 * 1000)
  }
})
