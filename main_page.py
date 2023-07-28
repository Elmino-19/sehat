import streamlit as st
import streamlit.components.v1 as components

def main():

  htm_string = '''
  <head>
    <meta charset="utf-8">
  </head>
  <body style="height: 100%; margin: 0">
    <div id="container" style="height: 100%"></div>


    <script type="text/javascript" src="https://fastly.jsdelivr.net/npm/echarts@5.4.2/dist/echarts.min.js"></script>
    <!-- Uncomment this line if you want to dataTool extension
    <script type="text/javascript" src="https://fastly.jsdelivr.net/npm/echarts@5.4.2/dist/extension/dataTool.min.js"></script>
    -->
    <!-- Uncomment this line if you want to use gl extension
    <script type="text/javascript" src="https://fastly.jsdelivr.net/npm/echarts-gl@2/dist/echarts-gl.min.js"></script>
    -->
    <!-- Uncomment this line if you want to echarts-stat extension
    <script type="text/javascript" src="https://fastly.jsdelivr.net/npm/echarts-stat@latest/dist/ecStat.min.js"></script>
    -->
    <!-- Uncomment this line if you want to use map
    <script type="text/javascript" src="https://fastly.jsdelivr.net/npm/echarts@4.9.0/map/js/china.js"></script>
    <script type="text/javascript" src="https://fastly.jsdelivr.net/npm/echarts@4.9.0/map/js/world.js"></script>
    -->
    <!-- Uncomment these two lines if you want to use bmap extension
    <script type="text/javascript" src="https://api.map.baidu.com/api?v=3.0&ak=YOUR_API_KEY"></script>
    <script type="text/javascript" src="https://fastly.jsdelivr.net/npm/echarts@5.4.2/dist/extension/bmap.min.js"></script>
    -->

    <script type="text/javascript">
      var dom = document.getElementById('container');
      var myChart = echarts.init(dom, null, {
        renderer: 'canvas',
        useDirtyRect: false
      });
      var app = {};

      var option;

      option = {
    graphic: {
      elements: [
        {
          type: 'text',
          left: 'center',
          top: 'center',
          style: {
            text: 'صحت',
            fontSize: 80,
            fontWeight: 'bold',
            lineDash: [0, 200],
            lineDashOffset: 0,
            fill: 'transparent',
            stroke: '#327fa8',
            lineWidth: 1
          },
          keyframeAnimation: {
            duration: 4000,
            loop: true,
            keyframes: [
              {
                percent: 0.7,
                style: {
                  fill: 'transparent',
                  lineDashOffset: 200,
                  lineDash: [200, 0]
                }
              },
              {
                // Stop for a while.
                percent: .7,
                style: {
                  fill: 'transparent'
                }
              },
              {
                percent: 1,
                style: {
                  fill: '#66ccff'
                }
              }
            ]
          }
        }
      ]
    }
  };

      if (option && typeof option === 'object') {
        myChart.setOption(option);
      }

      window.addEventListener('resize', myChart.resize);
    </script>
  </body>
  '''

  components.html(htm_string)  # JavaScript works

  st.title('اپلیکیشن کمک‌یار ورزشی')


  recorded_file = 'راهنمای برنامه.mp4'
  sample_vid = st.empty()
  sample_vid.video(recorded_file)
  st.subheader("راهنمای برنامه:")
  st.write("۱.ابتدا ویدیو مورد نظر خود را در برنامه آپلود کنید.")
  st.write("*توجه داشته باشید که که حرکت ورزشی باید به درستی در این ویدیو ترجیحا برای یک بار انجام شده باشد*")
  st.write("۲.بر روی گزینه ذخیره و نمایش زوایا کلیک کنید تا زوایای تحلیل شده ذخیره شود")
  st.write("۳.گزینه بررسی صحت حرکت را بزنید و روبه روی دوربین همانند ویدیویی که آپلود کردید شروع به ورزش کنید")
  
  st.write("این برنامه با استفاده از پردزاش تصویر در هوش مصنوعی وضعیت بدن افراد را هنگام انجام تمرینات ورزشی آنالیز می‌کند")
  st.info("ستایش سلطانی فرزانگان سه")
if __name__ == "__main__":
    main()