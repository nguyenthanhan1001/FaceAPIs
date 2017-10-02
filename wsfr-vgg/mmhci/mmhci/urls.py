from django.conf.urls import patterns, include, url
from django.contrib import admin
from django.conf.urls.static import static

import settings

admin.autodiscover()

urlpatterns = patterns('',
    url(r'^$', 'wsfr.views.hello'),
    url(r'^recognise$', 'wsfr.views.recognise'),
    url(r'^detect$', 'wsfr.views.detect'),
    #url(r'^admin/', include(admin.site.urls))
)

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)