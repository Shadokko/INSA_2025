### Adapted from https://www.thomasgrandjean.fr/portfolio/tutoriel/3/
import re
from branca.element import Template, MacroElement


__TEMPLATE__ = """
    {% macro html(this, kwargs) %}
    
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>jQuery UI Draggable - Default functionality</title>
      <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">
    
      <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
      <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
      
      <script>
      $( function() {
        $( "#maplegend" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                left: "auto",
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });
    
      </script>
    </head>
    <body>
    
    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
         border-radius:6px; padding: 10px; font-size:14px; left: 20px; bottom: 20px;'>

    <!--Changer ici pour mettre à jour le titre de la légende--> 
    <div class='legend-title'>MY_TITLE</div>
    <div class='legend-scale'>
      <ul class='legend-labels'>
        <!--Changer ici pour mettre à jour les éléments de la légende-->
        <li><span style='background:green;opacity:0.7;'></span>0-1</li>
        <li><span style='background:yellow;opacity:0.7;'></span>2-3</li>
        <li><span style='background:orange;opacity:0.7;'></span>4-5</li>
        <li><span style='background:red;opacity:0.7;'></span>6-7</li>
        <li><span style='background:darkviolet;opacity:0.7;'></span>8+</li>
      </ul>
    </div>
    </div>
     
    </body>
    </html>
    
    <style type='text/css'>
      /* Ensure container has a minimum width and visible overflow */
      .maplegend {
        min-width: 80px;
        color: #000;
        overflow: visible;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        color: #000 !important;
        display: block !important;
        clear: both;
      }
      .maplegend .legend-scale {
        overflow: visible;
      }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: none; /* avoid floating which can collapse/overflow */
        list-style: none;
      }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
      }
      .maplegend ul.legend-labels li span {
        display: inline-block;
        vertical-align: middle;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
      }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
      }
      .maplegend a {
        color: #777;
      }
    </style>
    {% endmacro %}"""


# TODO: make color legend replaceable with a kwarg, like title is.
def add_template2map(ma_carte, template=__TEMPLATE__, title="Atypicité"):
    """Add an HTML legend template to a folium Map.

    This function substitutes the placeholder title ("MY_TITLE") in the provided
    HTML template with the given ``title`` string, wraps the resulting HTML as a
    ``branca`` ``Template`` and ``MacroElement``, and attaches it to the root of
    the provided folium map.

    Parameters
    ----------
    ma_carte : folium.Map
        The folium Map instance to which the legend template will be added.
    template : str, optional
        HTML template string for the legend (default is :data:`__TEMPLATE__`).
    title : str, optional
        Title to substitute into the template in place of 'MY_TITLE' (default
        is ``"Atypicité"``).

    Returns
    -------
    folium.Map
        The same map instance with the legend template added as a MacroElement.

    Notes
    -----
    The function performs a simple string replacement of 'MY_TITLE' in the
    provided template, converts it to a :class:`branca.element.Template` wrapped
    in a :class:`branca.element.MacroElement`, and attaches it to the map root.
    """
    template = re.sub("MY_TITLE", title, template)
    macro = MacroElement()
    macro._template = Template(template)
    ma_carte.get_root().add_child(macro)

    return ma_carte
