#from django.db import models
from djongo import models

# Create your models here.


class Element(models.Model):
    name = models.CharField(max_length=2)
    #amount = models.IntegerField(default=1)

    def __str__(self):
        return self.name


class Space_Group(models.Model):
    name = models.TextField(help_text="P3m1")
    number = models.IntegerField()
    point_group = models.CharField(max_length=6, help_text="3m")
    crystal_system = models.TextField(help_text="trigonal")

    def __str__(self):
        return self.name

class UC_Element(models.Model):
    element = models.TextField()
    #num_element = models.IntegerField()

    def __str__(self):
        return self.element

    class meta:
        abstract = True
        ordering = ('element',)


#class Crystal(models.Model):

class Simulation(models.Model):

    #fields
    dir_name = models.TextField()
    #last_updated = models.DateTimeField()
    #unit_cell_formula = models.fields.ArrayFormField(model_form_class=Unit_Cell_Element)
    #unit_cell_formula = models.ManyToManyField(Unit_Cell_Element)
    #unit_cell_formula = models.TextField(help_text="C:2, Si:1, Hf:2, Al:1, Ti:1")
    unit_cell_formula=models.ManyToManyField(UC_Element)
    #completed_at = models.DateTimeField()
    #nsites = models.IntegerField()
    #chemsys = models.TextField(help_text="Al-C-Hf-Si-Ti")
    #elements = models.ManyToManyField(Element)
    #space_group = models.ForeignKey(Space_Group, on_delete=models.CASCADE)
    #run_type = models.TextField()
    def __str__(self):
        return self.dir_name

    class meta:
        ordering = ('dir_name',)

#class Unit_Cell_Formula(models.Model):
 #   unit_cell_formula = models.ListField(
  #      unit_cell_element = Unit_Cell_Element
   # )from django.contrib.postgres.fields import ArrayField