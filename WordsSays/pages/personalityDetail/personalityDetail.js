// pages/personalityDetail.js
Page({

  /**
   * 页面的初始数据
   */
  data: {
    showType: 'MBTI',
    detail:{
      type: 'INTP',
      typeDesc: 'THE ADVOCATE',
      note: 'Every man must decide whether he will walk in the light of creative altruism or in the darkness of destructive selfishness. \n --Martin Luther King',
      introduction: 'INFJs tend to see helping others as their purpose in life, but while people with this personality type can be found engaging rescue efforts and doing charity work, their real passion is to get to the heart of the issue so that people need not be rescued at all.',
      subsection:[
        { title: 'Live to Fight Another Day', desc: '   Really though, it is most important for INFJs to remember to take care of themselves. The passion of their convictions is perfectly capable of carrying them past their breaking point and if their zeal gets out of hand, they can find themselves exhausted, unhealthy and stressed. This becomes especially apparent when INFJs find themselves up against conflict and criticism – their sensitivity forces them to do everything they can to evade these seemingly personal attacks, but when the circumstances are unavoidable, they can fight back in highly irrational, unhelpful ways.'}
      ]
    }
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    this.data.showType = options.type;
    console.log(this.data.showType);

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