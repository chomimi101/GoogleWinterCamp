const app = getApp()
var hasClick = false;
var _famousPersons, _similarPersons;

var persons = JSON.parse(
  '{"data": [{ "name": "Mahatma Gandhi", "type": "INFJ", "type_description": "Advocate: Quiet and mystical, yet very inspiring and tireless idealists.", "description": "" },{ "name": "Warren Buffet", "type": "ISTJ", "type_description": "\u7269\u6d41\u5e08\uff1a\u5b9e\u9645\u4e14\u6ce8\u91cd\u4e8b\u5b9e\u7684\u4e2a\u4eba\uff0c\u53ef\u9760\u6027\u4e0d\u5bb9\u6000\u7591", "description": "" }, { "name": "Mother Theresa", "type": "ISFJ", "type_description": "\u5b88\u536b\u8005\uff1a\u975e\u5e38\u4e13\u6ce8\u800c\u6e29\u6696\u7684\u5b88\u62a4\u8005\uff0c\u65f6\u523b\u51c6\u5907\u7740\u4fdd\u62a4\u7231\u7740\u7684\u4eba\u4eec", "description": "" },  { "name": "Muck Zuckerberg", "type": "INTJ", "type_description": "\u5efa\u7b51\u5e08\uff1a\u5bcc\u6709\u60f3\u8c61\u529b\u548c\u6218\u7565\u6027\u7684\u601d\u60f3\u5bb6\uff0c\u4e00\u5207\u7686\u5728\u8ba1\u5212\u4e4b\u4e2d", "description": "" }, { "name": "Steve Jobs", "type": "ISTP", "type_description": "\u9274\u8d4f\u5bb6\uff1a\u5927\u80c6\u800c\u5b9e\u9645\u7684\u5b9e\u9a8c\u5bb6\uff0c\u64c5\u957f\u4f7f\u7528\u4efb\u4f55\u5f62\u5f0f\u7684\u5de5\u5177", "description": "" }, { "name": "Michael Jackson", "type": "ISFP", "type_description": "\u63a2\u9669\u5bb6\uff1a\u7075\u6d3b\u6709\u9b45\u529b\u7684\u827a\u672f\u5bb6\uff0c\u65f6\u523b\u51c6\u5907\u7740\u63a2\u7d22\u548c\u4f53\u9a8c\u65b0\u9c9c\u4e8b\u7269", "description": "" }, { "name": "J K Rowling", "type": "INFP", "type_description": "\u8c03\u505c\u8005\uff1a\u8bd7\u610f\uff0c\u5584\u826f\u7684\u5229\u4ed6\u4e3b\u4e49\u8005\uff0c\u603b\u662f\u70ed\u60c5\u5730\u4e3a\u6b63\u5f53\u7406\u7531\u63d0\u4f9b\u5e2e\u52a9", "description": "" }, { "name": "Jimmy Wales", "type": "INTP", "type_description": "\u903b\u8f91\u5b66\u5bb6\uff1a\u5177\u6709\u521b\u9020\u529b\u7684\u53d1\u660e\u5bb6\uff0c\u5bf9\u77e5\u8bc6\u6709\u7740\u6b62\u4e0d\u4f4f\u7684\u6e34\u671b", "description": "" }, { "name": "Donald Trump", "type": "ESTP", "type_description": "\u4f01\u4e1a\u5bb6\uff1a\u806a\u660e\uff0c\u7cbe\u529b\u5145\u6c9b\u5584\u4e8e\u611f\u77e5\u7684\u4eba\u4eec\uff0c\u771f\u5fc3\u4eab\u53d7\u751f\u6d3b\u5728\u8fb9\u7f18", "description": "" }, { "name": "Larry Ellison", "type": "ESFP", "type_description": "\u8868\u6f14\u8005\uff1a\u81ea\u53d1\u7684\uff0c\u7cbe\u529b\u5145\u6c9b\u800c\u70ed\u60c5\u7684\u8868\u6f14\u8005\uff0c\u751f\u6d3b\u5728\u4ed6\u4eec\u5468\u56f4\u6c38\u4e0d\u65e0\u804a", "description": "" }, { "name": "Walt Disney", "type": "ENFP", "type_description": "\u7ade\u9009\u8005\uff1a\u70ed\u60c5\uff0c\u6709\u521b\u9020\u529b\u7231\u793e\u4ea4\u7684\u81ea\u7531\u81ea\u5728\u7684\u4eba\uff0c\u603b\u80fd\u627e\u5230\u7406\u7531\u5fae\u7b11", "description": "" }, { "name": "Barack Obama", "type": "ENTP", "type_description": "\u8fa9\u8bba\u5bb6\uff1a\u806a\u660e\u597d\u5947\u7684\u601d\u60f3\u8005\uff0c\u4e0d\u4f1a\u653e\u5f03\u4efb\u4f55\u667a\u529b\u4e0a\u7684\u6311\u6218", "description": "" }, { "name": "Steve Ballmer", "type": "ESTJ", "type_description": "\u603b\u7ecf\u7406\uff1a\u51fa\u8272\u7684\u7ba1\u7406\u8005\uff0c\u5728\u7ba1\u7406\u4e8b\u60c5\u6216\u4eba\u7684\u65b9\u9762\u65e0\u4e0e\u4f26\u6bd4", "description": "" }, { "name": "Sam Walton", "type": "ESFJ", "type_description": "\u6267\u653f\u5b98\uff1a\u6781\u6709\u540c\u60c5\u5fc3\uff0c\u7231\u4ea4\u5f80\u53d7\u6b22\u8fce\u7684\u4eba\u4eec\uff0c\u603b\u662f\u70ed\u5fc3\u63d0\u4f9b\u5e2e\u52a9", "description": "" }, { "name": "Oprah Winfrey", "type": "ENFJ", "type_description": "\u4e3b\u4eba\u516c\uff1a\u5bcc\u6709\u9b45\u529b\u9f13\u821e\u4eba\u5fc3\u7684\u9886\u5bfc\u8005\uff0c\u6709\u4f7f\u542c\u4f17\u7740\u8ff7\u7684\u80fd\u529b", "description": "" }, { "name": "Bill Gates", "type": "ENTJ", "type_description": "\u6307\u6325\u5b98\uff1a\u5927\u80c6\uff0c\u5bcc\u6709\u60f3\u8c61\u529b\u4e14\u610f\u5fd7\u5f3a\u5927\u7684\u9886\u5bfc\u8005\uff0c\u603b\u80fd\u627e\u5230\u6216\u521b\u9020\u89e3\u51b3\u529e\u6cd5", "description": "" }]}')
// pages/index/index.js
Page({

  /**
   * 页面的初始数据
   */
  data: {
    userInfo: {},
    avatarUrl: './user-unlogin.png',
    posts: '',
    showResults: false,
    famousPersons:[
      {
        name: 'Trump', type: 'INFJ',
        desc: 'Trump is the presenident', avatar: '../../images/Donald Trump.jpg'
      },
    ],
    similarPersons:[],
    predict:{
      type: 'IMDT'
    }
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    //this.data.famousPersons = persons.data;
    // for(var i = 0; i < this.data.famousPersons.length){

    // }
    // for(var p in this.data.famousPersons){

    // }
    _famousPersons = []
    function preprocess(p){
      p['avatar'] = '../../images/' + p['name'] + '.jpg'
      _famousPersons.push(p)
    }
    persons.data.forEach(preprocess);

    console.log(_famousPersons);
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
    // wx.request({
    //   url: 'https://wordssays.aisspku.cn/',
    //   method:'GET',
    //   success: function(res){
    //     wx.hideLoading()
    //     _this.setData({ posts: JSON.stringify(res)})
    //   }
    // })
    //this.data.famousPersons = famousPersons
    setTimeout(function(){
      wx.hideLoading()
      _this.setData({ showResults: true })
      _this.setData({ famousPersons: _famousPersons })
      _this.setData({ predict: {type: "INFJ"}})
      //console.log(_this.data.predict.type)
      _this.data.famousPersons.forEach(function(p){
        console.log(p.type)
        console.log(_this.data.predict.type)
        if(p.type == _this.data.predict.type){
          _this.data.similarPersons.push(p);  
        }
      })
      console.log(_this.data.similarPersons)
      hasClick = false
    }, 1000)
    /*wx.request({
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

      })*/
      
  },

  showPersonalityDetail: function(){
    //wx.navigateTo({ url: '/pages/personalityDetail/personalityDetail'}) 
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