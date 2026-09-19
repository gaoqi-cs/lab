// pages/train/train.js
const app = getApp()

Page({
  data: {
    imageList: [],
    label: '',
    clearOld: false,
    isTraining: false,
    status: {status: 'idle', result: null},
    datasetClasses: [],
    datasetTotal: 0,
    ws: null,
    canUpload: false,
    canTrain: false
  },

  onLoad() {
    this.initWebSocket()
    this.loadDataset()
    this.updateActions()
  },

  onShow() {
    this.loadDataset()
    this.updateActions()
  },

  initWebSocket() {
    const base = app.globalData.apiBase
      .replace('http://', '')
      .replace('https://', '')
    const ws = wx.connectSocket({url: `ws://${base}/api/train/ws`})

    ws.onMessage((res) => {
      try {
        const data = JSON.parse(res.data)
        this.setData({
          status: data,
          isTraining: data.status === 'running'
        }, this.updateActions)
      } catch (error) {
        console.error('训练状态解析失败', error)
      }
    })

    ws.onClose(() => {
      if (this.data.ws !== ws) return
      this.reconnectTimer = setTimeout(() => this.initWebSocket(), 3000)
    })

    ws.onError(() => {
      console.warn('训练状态连接失败，请确认后端已启动')
    })
    this.setData({ws})
  },

  loadDataset() {
    wx.request({
      url: `${app.globalData.apiBase}/api/train/dataset`,
      success: (res) => {
        if (res.statusCode !== 200) return
        this.setData({
          datasetClasses: res.data.classes || [],
          datasetTotal: res.data.total || 0
        }, this.updateActions)
      },
      fail: () => {
        console.warn('训练样本统计读取失败，请确认后端已启动')
      }
    })
  },

  deleteDatasetClass(event) {
    const label = event.currentTarget.dataset.name
    if (this.data.isTraining) return

    wx.showModal({
      title: '删除训练案例',
      content: `确定删除“${label}”及其全部样本图片吗？`,
      confirmColor: '#E65C3C',
      success: (modal) => {
        if (!modal.confirm) return
        wx.showLoading({title: '正在删除...', mask: true})
        wx.request({
          url: `${app.globalData.apiBase}/api/train/dataset/delete`,
          method: 'POST',
          data: {label},
          success: (res) => {
            if (res.statusCode === 200) {
              wx.showToast({title: '训练案例已删除', icon: 'success'})
              this.loadDataset()
            } else {
              wx.showToast({title: res.data.detail || '删除失败', icon: 'none'})
            }
          },
          fail: () => wx.showToast({title: '无法连接后端服务', icon: 'none'}),
          complete: () => wx.hideLoading()
        })
      }
    })
  },

  updateActions() {
    const hasBatch = this.data.imageList.length > 0
      && this.data.label.trim() !== ''
    const hasDataset = this.data.datasetClasses.length >= 2
      && this.data.datasetClasses.every(item => item.count >= 2)
    this.setData({
      canUpload: hasBatch && !this.data.isTraining,
      canTrain: hasDataset && !this.data.isTraining
    })
  },

  chooseImages() {
    const remaining = 9 - this.data.imageList.length
    if (remaining <= 0) {
      wx.showToast({title: '本批最多 9 张，请先保存本批', icon: 'none'})
      return
    }

    wx.chooseMedia({
      count: Math.min(9, remaining),
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      success: (res) => {
        const newFiles = res.tempFiles.map(file => ({
          path: file.tempFilePath,
          name: file.tempFilePath.split('/').pop() || 'image.jpg',
          size: file.size
        }))
        this.setData({
          imageList: [...this.data.imageList, ...newFiles.slice(0, remaining)]
        }, this.updateActions)
      }
    })
  },

  clearImages() {
    this.setData({imageList: []}, this.updateActions)
  },

  removeImage(event) {
    const imageList = [...this.data.imageList]
    imageList.splice(event.currentTarget.dataset.index, 1)
    this.setData({imageList}, this.updateActions)
  },

  onLabelInput(event) {
    this.setData({label: event.detail.value}, this.updateActions)
  },

  onClearOldChange() {
    this.setData({clearOld: !this.data.clearOld}, this.updateActions)
  },

  uploadBatch() {
    if (!this.data.canUpload) return

    this.setData({isTraining: true}, this.updateActions)
    wx.showLoading({title: '正在保存样本...', mask: true})
    const fs = wx.getFileSystemManager()
    const tasks = this.data.imageList.map(item => new Promise((resolve, reject) => {
      fs.readFile({
        filePath: item.path,
        success: (res) => {
          const base64 = wx.arrayBufferToBase64(res.data)
          const ext = item.name.split('.').pop() || 'jpg'
          const mimeType = `image/${ext === 'jpg' ? 'jpeg' : ext}`
          resolve(`data:${mimeType};base64,${base64}`)
        },
        fail: reject
      })
    }))

    Promise.all(tasks).then(images => {
      wx.request({
        url: `${app.globalData.apiBase}/api/train/finetune`,
        method: 'POST',
        data: {
          images,
          label: this.data.label.trim(),
          clearOld: this.data.clearOld
        },
        success: (res) => {
          if (res.statusCode !== 202) {
            wx.showToast({title: res.data.detail || '样本保存失败', icon: 'none'})
            return
          }
          const counts = res.data.class_counts || {}
          this.setData({
            imageList: [],
            label: '',
            clearOld: false,
            isTraining: false,
            datasetClasses: Object.keys(counts).map(name => ({name, count: counts[name]})),
            datasetTotal: Object.values(counts).reduce((total, count) => total + count, 0)
          }, this.updateActions)
          wx.showToast({title: '本批样本已保存', icon: 'success'})
        },
        fail: () => wx.showToast({title: '无法连接后端服务', icon: 'none'}),
        complete: () => {
          wx.hideLoading()
          this.setData({isTraining: false}, this.updateActions)
        }
      })
    }).catch(() => {
      wx.hideLoading()
      wx.showToast({title: '图片读取/转换失败', icon: 'none'})
      this.setData({isTraining: false}, this.updateActions)
    })
  },

  startTraining() {
    if (!this.data.canTrain) return
    this.setData({isTraining: true}, this.updateActions)
    wx.showLoading({title: '正在启动训练...', mask: true})
    wx.request({
      url: `${app.globalData.apiBase}/api/train/start`,
      method: 'POST',
      success: (res) => {
        if (res.statusCode === 200) {
          this.setData({status: {status: 'running', result: null}})
          wx.showToast({title: '训练已启动', icon: 'success'})
        } else {
          wx.showToast({title: res.data.detail || '训练启动失败', icon: 'none'})
          this.setData({isTraining: false}, this.updateActions)
        }
      },
      fail: () => {
        wx.showToast({title: '无法连接后端服务', icon: 'none'})
        this.setData({isTraining: false}, this.updateActions)
      },
      complete: () => wx.hideLoading()
    })
  },

  stopTraining() {
    if (this.data.status.status !== 'running') return
    wx.request({
      url: `${app.globalData.apiBase}/api/train/stop`,
      method: 'POST',
      success: (res) => {
        if (res.statusCode === 200) {
          wx.showToast({title: '已请求停止训练', icon: 'success'})
          this.setData({isTraining: false}, this.updateActions)
        } else {
          wx.showToast({title: res.data.detail || '停止训练失败', icon: 'none'})
        }
      },
      fail: () => wx.showToast({title: '停止训练失败', icon: 'none'})
    })
  },

  onUnload() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    if (this.data.ws) this.data.ws.close()
  }
})
