// pages/personalityDetail.js
Page({

  /**
   * 页面的初始数据
   */
  data: {
    showType: 'MBTI',
    detail:{
      type: 'INTP',
      typeDesc: '逻辑学家',
      note: '借鉴昨天，活在今天，憧憬明天。 重要的是绝不停止质疑。--Albert Einstein',
      introduction: '只有 3% 的人口为逻辑学家型人格，极为罕见，尽管如此，他们也并不以为意，因为他们根本不屑与“平庸”为伍。 逻辑学家们展现出积极主动的创造性，异于常人的视角以及永不枯竭的智慧，这都令他们深感自豪。 人们常常将逻辑学家称为哲学家、思考者，或是爱空想的教授，在历史的长河中，许多科学发现就是他们的智慧之花结出的丰硕果实。',
      subsection:[
        { title: '混混噩噩的生活不值得过', desc: '具有逻辑学家人格类型的人热衷于各种模式，而发现话语之间的纰漏几乎是他们与生俱来的习惯，所以，对逻辑学家说谎可不是什么明智之举。 颇有讽刺意味的是，逻辑学家的话却不可全信 — 不是因为他们不够诚实，而是因为逻辑学家人格类型的人喜欢在跟自己的辩论中分享并未完全成熟的想法，他们只是为了从他人口中试探对各种想法和理论的意见，而不是将他们作为真正的谈话伙伴。'}
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