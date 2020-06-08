from django.contrib import admin
#from .models import Entry
from .materials.entry import Entry
from .data.meta_data import MetaData
from simulation.analysis.vasp.calculation import Calculation
#from django_better_admin_arrayfield.admin.mixins import DynamicArrayMixin
# Register your models here.

class MyModelAdmin(admin.ModelAdmin):
#    admin.site.register(UC_Element)
#    admin.site.register(Simulation)
    admin.site.register(Entry)
    admin.site.register(MetaData)
    admin.site.register(Calculation)
