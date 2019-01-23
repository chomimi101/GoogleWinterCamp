const app = getApp()
var hasClick = false;
// pages/index/index.js
Page({

  /**
   * 页面的初始数据
   */
  data: {
    userInfo: {},
    avatarUrl: './user-unlogin.png',
    posts: '',
    showResults: false
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    // 获取用户信息
    wx.getSetting({
      success: res => {
        if (res.authSetting['scope.userInfo']) {
          // 已经授权，可以直接调用 getUserInfo 获取头像昵称，不会弹框
          wx.getUserInfo({
            success: res => {
              this.setData({
                avatarUrl: res.userInfo.avatarUrl,
                userInfo: res.userInfo
              })
            }
          })
        }
      }
    })
  },

  onGetUserInfo: function (e) {
    if (!this.logged && e.detail.userInfo) {
      this.setData({
        logged: true,
        avatarUrl: e.detail.userInfo.avatarUrl,
        userInfo: e.detail.userInfo
      })
    }
  },

  getDataBindTap: function (e) {
    var result = e.detail.value;
    console.log(result)
   
      // wx.request({
      //   url: 'http://127.0.0.1:5000/predictPersonality',
      //   method: 'POST',
      //   header: { 'content-type': 'application/json' },
      //   data: {
      //     posts: result
      //   },
      //   success: function (res) {
      //     console.log(res)// 服务器回包信息

      //   }

      // })
  },
  startPredict: function(){
    if (hasClick) {
      return;
    }
    this.data.showResults = false
    console.log(this.data.showResults)
    hasClick = true;
    wx.showLoading();
    console.log(this.data.posts);
    var _this = this
    wx.request({
        url: 'http://127.0.0.1:5000/predictPersonality',
        method: 'POST',
        header: { 'content-type': 'application/json' },
        data: {
          posts: this.data.posts
        },
        success: function (res) {
          if (res.statusCode === 200) {
            console.log(res.data)// 服务器回包内容
          }
        },
      fail: function (res) {

        wx.showToast({ title: '系统错误' })

      },

      complete: function (res) {

        wx.hideLoading()
        _this.setData({showResults : true})
        hasClick = false

      }

      })
      
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  }
})