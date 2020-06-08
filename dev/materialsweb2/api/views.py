from rest_framework import viewsets
from django.http import HttpResponseNotFound, JsonResponse, HttpResponse
from .serializers import ElementSerializer
from simulation.materials.element import Element
import datetime

class ElementViewSet(viewsets.ModelViewSet):
    queryset = Element.objects.all().order_by('name')
    serializer_class = ElementSerializer
    
    
def rest(request):
    """
    Render JSON data for the REST API.
    """
    query_components = request.get_full_path().split('/')

    api_key = query_components[-1]
    query = query_components[query_components.index('materials')+1]

    only_ids = False
    if query_components[-2] == "ids":
        only_ids = True

    if "all" in query:
        query = '{}'

    if 'hse' in query_components or 'HSE' in query_components:
        run_type = 'HF'
    else:
        run_type = 'GGA'

    valid_api_keys = [u.last_name for u in User.objects.all()]
    api_key = query_components[-1]
    if api_key in valid_api_keys:
        if only_ids:
            properties = get_ids(query, 'mat2d', run_type=run_type)
            data = {
                "valid_response": True,
                "created_at": datetime.datetime.now().strftime(
                        '%Y-%m-%d-%H-%M-%S-%f'
                    ),
                    'response': properties
                }
        else:
            properties = get_all_properties(query, 'mat2d', run_type=run_type)
            if len(properties):
                if query != '{}':
                    response = properties[0]
                    response['last_updated'] = response['last_updated'].strftime(
                        '%Y-%m-%d')
                    print( response['calculations'][-1]['output'])
                    response['_id'] = str(response['_id'])
                    response['final_structure'] = Structure.from_dict(
                        response['calculations'][-1]['output']['crystal']
                    ).to(fmt='json')
                    data = {
                        'valid_response': True,
                        'created_at': datetime.datetime.now().strftime(
                            '%Y-%m-%d-%H-%M-%S-%f'
                        ),
                        'response': [response]
                    }
                else:
                    responses = []
                    for entry in [p for p in properties if p['calculations'][0]\
                            ['input']['kpoints']['generation_style'] in ['Monkhorst',
                            'Gamma']]:
                        response = entry
                        response['last_updated'] = response['last_updated'].strftime(
                            '%Y-%m-%d')
                        response['_id'] = str(response['_id'])
                        response['final_structure'] = Structure.from_dict(
                            response['calculations'][-1]['output']['crystal']
                        ).to(fmt='json')
                        responses.append(response)
                    data = {
                        'valid_response': True,
                        'created_at': datetime.datetime.now().strftime(
                            '%Y-%m-%d-%H-%M-%S-%f'
                        ),
                        'response': responses
                    }

            else:
                data = {'valid_response': False}
    else:
        data = {'valid_response': False, 'Error': 'Invalid API key supplied.'}

    return JsonResponse(data)
