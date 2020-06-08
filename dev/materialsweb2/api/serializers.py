from rest_framework import serializers
from simulation.materials.element import Element


class ElementSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Element
        fields = '__all__'
