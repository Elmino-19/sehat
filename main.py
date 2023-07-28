# به منظور اجرای برنامه Streamlit، شما باید ابتدا کتابخانه‌های مورد نیاز را نصب کنید. برای این کار، می‌توانید از فایل requirements.txt استفاده کنید. در ترمینال، به مسیر پروژه خود بروید و دستور زیر را وارد کنید:
#pip install -r requirements.txt
# این دستور، تمام کتابخانه‌های مورد نیاز را بر اساس فایل requirements.txt نصب خواهد کرد.
#
# سپس، برای اجرای برنامه، باید دستور زیر را در ترمینال وارد کنید:
#
#  streamlit run main.py
#
#در نهایت برنامه بر روی مرورگر شما قابل اجرا است


import streamlit as st
import streamlit.components.v1 as components
import hydralit_components as hc


#make it look nice from the start

st.set_page_config(layout='wide',initial_sidebar_state='collapsed')

with open('pages/static/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# specify the primary menu definition
menu_data = [
    {'icon': "fas fa-file-upload", 'label':"بارگذاری ویدیو"},
    {'icon': "fas fa-camera", 'label':"بررسی صحت حرکت"}]
over_theme = {'txc_inactive': '#FFFFFF'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='صفحه اصلی',
    login_name='ورود/خروج',
    hide_streamlit_markers=True, #will show the st hamburger as well as the navbar now!
    sticky_nav=False, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)

#get the id of the menu item clicked
st.info(f"{menu_id}")

if menu_id == 'ورود/خروج':
    from authentication import main
    main()
    
elif menu_id == 'صفحه اصلی':
    from main_page import main
    main()
 
elif menu_id == "بارگذاری ویدیو":
    from upload_video import main
    main()
 
elif menu_id == "بررسی صحت حرکت":
    from camera import main 
    main()