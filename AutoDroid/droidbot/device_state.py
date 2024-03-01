import copy
import math
import os
import pdb

import tools
from .utils import md5
from .input_event import TouchEvent, LongTouchEvent, ScrollEvent, SetTextEvent, KeyEvent, UIEvent
import hashlib
from treelib import Tree
import networkx as nx
import numpy as np

class DeviceState(object):
    """
    the state of the current device
    """

    def __init__(self, device, views, foreground_activity, activity_stack, background_services,
                 tag=None, screenshot_path=None):
        self.device = device
        self.foreground_activity = foreground_activity
        self.activity_stack = activity_stack if isinstance(activity_stack, list) else []
        self.background_services = background_services
        if tag is None:
            from datetime import datetime
            tag = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.tag = tag
        self.screenshot_path = screenshot_path
        self.views = self.__parse_views(views)
        
        self.bk_views = copy.deepcopy(self.views)
        self.view_graph = self._build_view_graph()
        # self._adjust_view_clickability()

        self.view_tree = {}
        self.__assemble_view_tree(self.view_tree, self.views)
        self.__generate_view_strs()
        self.state_str = self.__get_hashed_state_str()
        self.structure_str = self.__get_content_free_state_str()
        self.search_content = self.__get_search_content()
        self.possible_events = None
        self.width = device.get_width(refresh=True)
        self.height = device.get_height(refresh=False)
        self._save_important_view_ids()
        

    @property
    def activity_short_name(self):
        return self.foreground_activity.split('.')[-1]
    
        
    def _save_important_view_ids(self):
        _, _, _, important_view_ids = self.get_described_actions(remove_time_and_ip=False)
        ids_path = self.device.output_dir +'/states_view_ids'
        if not os.path.exists(ids_path):
            os.mkdir(ids_path)
        # if not isinstance(current_state, str):
        #     current_state_str = current_state.state_str
        # else:
        #     current_state_str = current_state
        important_view_id_path = self.device.output_dir +'/states_view_ids/'+ self.state_str + '.txt'
        f = open(important_view_id_path, 'w')
        f.write(str(important_view_ids))
        f.close()

    def __get_hashed_state_str(self):
        state, _, _, _ = self.get_described_actions(remove_time_and_ip=True)
        hashed_string = tools.hash_string(state)
        return hashed_string

    def to_dict(self):
        state = {'tag': self.tag,
                 'state_str': self.state_str,
                 'state_str_content_free': self.structure_str,
                 'foreground_activity': self.foreground_activity,
                 'activity_stack': self.activity_stack,
                 'background_services': self.background_services,
                 'width': self.width,
                 'height': self.height,
                 'views': self.views}
        return state

    def to_json(self):
        import json
        return json.dumps(self.to_dict(), indent=2)

    def __parse_views(self, raw_views):
        views = []
        if not raw_views or len(raw_views) == 0:
            return views

        for view_dict in raw_views:
            # # Simplify resource_id
            # resource_id = view_dict['resource_id']
            # if resource_id is not None and ":" in resource_id:
            #     resource_id = resource_id[(resource_id.find(":") + 1):]
            #     view_dict['resource_id'] = resource_id
            views.append(view_dict)
        return views

    def __assemble_view_tree(self, root_view, views):
        if not len(self.view_tree): # bootstrap
            self.view_tree = copy.deepcopy(views[0])
            self.__assemble_view_tree(self.view_tree, views)
        else:
            children = list(enumerate(root_view["children"]))
            if not len(children):
                return
            for i, j in children:
                root_view["children"][i] = copy.deepcopy(self.views[j])
                self.__assemble_view_tree(root_view["children"][i], views)

    def __generate_view_strs(self):
        for view_dict in self.views:
            self.__get_view_str(view_dict)
            # self.__get_view_structure(view_dict)

    @staticmethod
    def __calculate_depth(views):
        root_view = None
        for view in views:
            if DeviceState.__safe_dict_get(view, 'parent') == -1:
                root_view = view
                break
        DeviceState.__assign_depth(views, root_view, 0)

    @staticmethod
    def __assign_depth(views, view_dict, depth):
        view_dict['depth'] = depth
        for view_id in DeviceState.__safe_dict_get(view_dict, 'children', []):
            DeviceState.__assign_depth(views, views[view_id], depth + 1)

    def __get_state_str(self):
        state_str_raw = self.__get_state_str_raw()
        return md5(state_str_raw)

    def __get_state_str_raw(self):
        if self.device.humanoid is not None:
            import json
            from xmlrpc.client import ServerProxy
            proxy = ServerProxy("http://%s/" % self.device.humanoid)
            return proxy.render_view_tree(json.dumps({
                "view_tree": self.view_tree,
                "screen_res": [self.device.display_info["width"],
                               self.device.display_info["height"]]
            }))
        else:
            view_signatures = set()
            for view in self.views:
                view_signature = DeviceState.__get_view_signature(view)
                if view_signature:
                    view_signatures.add(view_signature)
            return "%s{%s}" % (self.foreground_activity, ",".join(sorted(view_signatures)))

    def __get_content_free_state_str(self):
        if self.device.humanoid is not None:
            import json
            from xmlrpc.client import ServerProxy
            proxy = ServerProxy("http://%s/" % self.device.humanoid)
            state_str = proxy.render_content_free_view_tree(json.dumps({
                "view_tree": self.view_tree,
                "screen_res": [self.device.display_info["width"],
                               self.device.display_info["height"]]
            }))
        else:
            view_signatures = set()
            for view in self.views:
                view_signature = DeviceState.__get_content_free_view_signature(view)
                if view_signature:
                    view_signatures.add(view_signature)
            state_str = "%s{%s}" % (self.foreground_activity, ",".join(sorted(view_signatures)))
        import hashlib
        return hashlib.md5(state_str.encode('utf-8')).hexdigest()

    def __get_search_content(self):
        """
        get a text for searching the state
        :return: str
        """
        words = [",".join(self.__get_property_from_all_views("resource_id")),
                 ",".join(self.__get_property_from_all_views("text"))]
        return "\n".join(words)

    def __get_property_from_all_views(self, property_name):
        """
        get the values of a property from all views
        :return: a list of property values
        """
        property_values = set()
        for view in self.views:
            property_value = DeviceState.__safe_dict_get(view, property_name, None)
            if property_value:
                property_values.add(property_value)
        return property_values

    def save2dir(self, output_dir=None):
        try:
            if output_dir is None:
                if self.device.output_dir is None:
                    return
                else:
                    output_dir = os.path.join(self.device.output_dir, "states")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            dest_state_json_path = "%s/state_%s.json" % (output_dir, self.tag)
            if self.device.adapters[self.device.minicap]:
                dest_screenshot_path = "%s/screen_%s.jpg" % (output_dir, self.tag)
            else:
                dest_screenshot_path = "%s/screen_%s.png" % (output_dir, self.tag)
            state_json_file = open(dest_state_json_path, "w")
            state_json_file.write(self.to_json())
            state_json_file.close()
            import shutil
            shutil.copyfile(self.screenshot_path, dest_screenshot_path)
            self.screenshot_path = dest_screenshot_path
            # from PIL.Image import Image
            # if isinstance(self.screenshot_path, Image):
            #     self.screenshot_path.save(dest_screenshot_path)
        except Exception as e:
            self.device.logger.warning(e)

    def save_view_img(self, view_dict, output_dir=None):
        try:
            if output_dir is None:
                if self.device.output_dir is None:
                    return
                else:
                    output_dir = os.path.join(self.device.output_dir, "views")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            view_str = view_dict['view_str']
            if self.device.adapters[self.device.minicap]:
                view_file_path = "%s/view_%s.jpg" % (output_dir, view_str)
            else:
                view_file_path = "%s/view_%s.png" % (output_dir, view_str)
            if os.path.exists(view_file_path):
                return
            from PIL import Image
            # Load the original image:
            view_bound = view_dict['bounds']
            original_img = Image.open(self.screenshot_path)
            # view bound should be in original image bound
            view_img = original_img.crop((min(original_img.width - 1, max(0, view_bound[0][0])),
                                          min(original_img.height - 1, max(0, view_bound[0][1])),
                                          min(original_img.width, max(0, view_bound[1][0])),
                                          min(original_img.height, max(0, view_bound[1][1]))))
            view_img.convert("RGB").save(view_file_path)
        except Exception as e:
            self.device.logger.warning(e)

    def is_different_from(self, another_state):
        """
        compare this state with another
        @param another_state: DeviceState
        @return: boolean, true if this state is different from other_state
        """
        return self.state_str != another_state.state_str

    @staticmethod
    def __get_view_signature(view_dict):
        """
        get the signature of the given view
        @param view_dict: dict, an element of list DeviceState.views
        @return:
        """
        if 'signature' in view_dict:
            return view_dict['signature']

        view_text = DeviceState.__safe_dict_get(view_dict, 'text', "None")
        if view_text is None or len(view_text) > 50:
            view_text = "None"

        signature = "[class]%s[resource_id]%s[text]%s[%s,%s,%s]" % \
                    (DeviceState.__safe_dict_get(view_dict, 'class', "None"),
                     DeviceState.__safe_dict_get(view_dict, 'resource_id', "None"),
                     view_text,
                     DeviceState.__key_if_true(view_dict, 'enabled'),
                     DeviceState.__key_if_true(view_dict, 'checked'),
                     DeviceState.__key_if_true(view_dict, 'selected'))
        view_dict['signature'] = signature
        return signature

    @staticmethod
    def __get_content_free_view_signature(view_dict):
        """
        get the content-free signature of the given view
        @param view_dict: dict, an element of list DeviceState.views
        @return:
        """
        if 'content_free_signature' in view_dict:
            return view_dict['content_free_signature']
        content_free_signature = "[class]%s[resource_id]%s" % \
                                 (DeviceState.__safe_dict_get(view_dict, 'class', "None"),
                                  DeviceState.__safe_dict_get(view_dict, 'resource_id', "None"))
        view_dict['content_free_signature'] = content_free_signature
        return content_free_signature

    def __get_view_str(self, view_dict):
        """
        get a string which can represent the given view
        @param view_dict: dict, an element of list DeviceState.views
        @return:
        """
        if 'view_str' in view_dict:
            return view_dict['view_str']
        view_signature = DeviceState.__get_view_signature(view_dict)
        parent_strs = []
        for parent_id in self.get_all_ancestors(view_dict):
            parent_strs.append(DeviceState.__get_view_signature(self.views[parent_id]))
        parent_strs.reverse()
        child_strs = []
        for child_id in self.get_all_children(view_dict):
            child_strs.append(DeviceState.__get_view_signature(self.views[child_id]))
        child_strs.sort()
        view_str = "Activity:%s\nSelf:%s\nParents:%s\nChildren:%s" % \
                   (self.foreground_activity, view_signature, "//".join(parent_strs), "||".join(child_strs))
        import hashlib
        view_str = hashlib.md5(view_str.encode('utf-8')).hexdigest()
        view_dict['view_str'] = view_str
        return view_str

    def __get_view_structure(self, view_dict):
        """
        get the structure of the given view
        :param view_dict: dict, an element of list DeviceState.views
        :return: dict, representing the view structure
        """
        if 'view_structure' in view_dict:
            return view_dict['view_structure']
        width = DeviceState.get_view_width(view_dict)
        height = DeviceState.get_view_height(view_dict)
        class_name = DeviceState.__safe_dict_get(view_dict, 'class', "None")
        children = {}

        root_x = view_dict['bounds'][0][0]
        root_y = view_dict['bounds'][0][1]

        child_view_ids = self.__safe_dict_get(view_dict, 'children')
        if child_view_ids:
            for child_view_id in child_view_ids:
                child_view = self.views[child_view_id]
                child_x = child_view['bounds'][0][0]
                child_y = child_view['bounds'][0][1]
                relative_x, relative_y = child_x - root_x, child_y - root_y
                children["(%d,%d)" % (relative_x, relative_y)] = self.__get_view_structure(child_view)

        view_structure = {
            "%s(%d*%d)" % (class_name, width, height): children
        }
        view_dict['view_structure'] = view_structure
        return view_structure

    @staticmethod
    def __key_if_true(view_dict, key):
        return key if (key in view_dict and view_dict[key]) else ""

    @staticmethod
    def __safe_dict_get(view_dict, key, default=None):
        return_itm = view_dict[key] if (key in view_dict) else default
        if return_itm == None:
            return_itm = ''
        return return_itm
    
    @staticmethod
    def get_view_center(view_dict):
        """
        return the center point in a view
        @param view_dict: dict, an element of DeviceState.views
        @return: a pair of int
        """
        bounds = view_dict['bounds']
        return (bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2

    @staticmethod
    def get_view_width(view_dict):
        """
        return the width of a view
        @param view_dict: dict, an element of DeviceState.views
        @return: int
        """
        bounds = view_dict['bounds']
        return int(math.fabs(bounds[0][0] - bounds[1][0]))

    @staticmethod
    def get_view_height(view_dict):
        """
        return the height of a view
        @param view_dict: dict, an element of DeviceState.views
        @return: int
        """
        bounds = view_dict['bounds']
        return int(math.fabs(bounds[0][1] - bounds[1][1]))

    def get_all_ancestors(self, view_dict):
        """
        Get temp view ids of the given view's ancestors
        :param view_dict: dict, an element of DeviceState.views
        :return: list of int, each int is an ancestor node id
        """
        result = []
        parent_id = self.__safe_dict_get(view_dict, 'parent', -1)
        if 0 <= parent_id < len(self.views):
            result.append(parent_id)
            result += self.get_all_ancestors(self.views[parent_id])
        return result

    def get_all_children(self, view_dict):
        """
        Get temp view ids of the given view's children
        :param view_dict: dict, an element of DeviceState.views
        :return: set of int, each int is a child node id
        """
        children = self.__safe_dict_get(view_dict, 'children')
        if not children:
            return set()
        children = set(children)
        for child in children:
            children_of_child = self.get_all_children(self.views[child])
            children.union(children_of_child)
        return children

    def get_app_activity_depth(self, app):
        """
        Get the depth of the app's activity in the activity stack
        :param app: App
        :return: the depth of app's activity, -1 for not found
        """
        depth = 0
        for activity_str in self.activity_stack:
            if app.package_name in activity_str:
                return depth
            depth += 1
        return -1

    def get_possible_input(self):
        """
        Get a list of possible input events for this state
        :return: list of InputEvent
        """
        if self.possible_events:
            return [] + self.possible_events
        possible_events = []
        enabled_view_ids = []
        touch_exclude_view_ids = set()
        for view_dict in self.views:
            # exclude navigation bar if exists
            if self.__safe_dict_get(view_dict, 'enabled') and \
                    self.__safe_dict_get(view_dict, 'visible') and \
                    self.__safe_dict_get(view_dict, 'resource_id') not in \
               ['android:id/navigationBarBackground',
                'android:id/statusBarBackground']:
                enabled_view_ids.append(view_dict['temp_id'])
        # enabled_view_ids.reverse()

        for view_id in enabled_view_ids:
            if self.__safe_dict_get(self.views[view_id], 'clickable'):
                possible_events.append(TouchEvent(view=self.views[view_id]))
                touch_exclude_view_ids.add(view_id)
                touch_exclude_view_ids.union(self.get_all_children(self.views[view_id]))

        for view_id in enabled_view_ids:
            if self.__safe_dict_get(self.views[view_id], 'scrollable'):
                possible_events.append(ScrollEvent(view=self.views[view_id], direction="UP"))
                possible_events.append(ScrollEvent(view=self.views[view_id], direction="DOWN"))
                possible_events.append(ScrollEvent(view=self.views[view_id], direction="LEFT"))
                possible_events.append(ScrollEvent(view=self.views[view_id], direction="RIGHT"))

        for view_id in enabled_view_ids:
            if self.__safe_dict_get(self.views[view_id], 'checkable'):
                possible_events.append(TouchEvent(view=self.views[view_id]))
                touch_exclude_view_ids.add(view_id)
                touch_exclude_view_ids.union(self.get_all_children(self.views[view_id]))

        for view_id in enabled_view_ids:
            if self.__safe_dict_get(self.views[view_id], 'long_clickable'):
                possible_events.append(LongTouchEvent(view=self.views[view_id]))

        for view_id in enabled_view_ids:
            if self.__safe_dict_get(self.views[view_id], 'editable'):
                possible_events.append(SetTextEvent(view=self.views[view_id], text="HelloWorld"))
                touch_exclude_view_ids.add(view_id)
                # TODO figure out what event can be sent to editable views
                pass

        # for view_id in enabled_view_ids:
        #     if view_id in touch_exclude_view_ids:
        #         continue
        #     children = self.__safe_dict_get(self.views[view_id], 'children')
        #     if children and len(children) > 0:
        #         continue
        #     possible_events.append(TouchEvent(view=self.views[view_id]))

        # For old Android navigation bars
        # possible_events.append(KeyEvent(name="MENU"))

        self.possible_events = possible_events
        return [] + possible_events
    
    def _get_self_ancestors_property(self, view, key, default=None):
        all_views = [view] + [self.views[i] for i in self.get_all_ancestors(view)]
        for v in all_views:
            value = self.__safe_dict_get(v, key)
            if value:
                return value
        return default
    

    def _merge_text(self, view_text, content_description):
        text = ''
        if view_text:
            view_text = view_text.replace('\n', '  ')
            view_text = f'{view_text[:20]}...' if len(view_text) > 20 else view_text
            text += view_text
            text += ' '
        if content_description:
            content_description = content_description.replace('\n', '  ')
            content_description = f'{content_description[:20]}...' if len(content_description) > 20 else content_description
            text += content_description
        return text
        
    def _remove_view_ids(self, views):
        import re
        removed_views = []
        for view_desc in views:
            view_desc_without_id = tools.get_view_without_id(view_desc)
            removed_views.append(view_desc_without_id)
        return removed_views

    def get_described_actions_bk(self, prefix=''):
        """
        Get a text description of current state
        """
        # import pdb;pdb.set_trace()
        enabled_view_ids = []
        for view_dict in self.views:
            # exclude navigation bar if exists
            if self.__safe_dict_get(view_dict, 'visible') and \
                self.__safe_dict_get(view_dict, 'resource_id') not in \
               ['android:id/navigationBarBackground',
                'android:id/statusBarBackground']:
                enabled_view_ids.append(view_dict['temp_id'])
        
        text_frame = "<p id=@ class='&'>#</p>"
        btn_frame = "<button id=@ class='&' checked=$>#</button>"
        input_frame = "<input id=@ class='&' >#</input>"
        scroll_down_frame = "<div id=@ class='scroller'>scroll down</div>"
        scroll_up_frame = "<div id=@ class='scroller'>scroll up</div>"

        view_descs = []
        available_actions = []
        for view_id in enabled_view_ids:
            view = self.views[view_id]
            clickable = self._get_self_ancestors_property(view, 'clickable')
            scrollable = self.__safe_dict_get(view, 'scrollable')
            checkable = self._get_self_ancestors_property(view, 'checkable')
            long_clickable = self._get_self_ancestors_property(view, 'long_clickable')
            editable = self.__safe_dict_get(view, 'editable')
            actionable = clickable or scrollable or checkable or long_clickable or editable
            checked = self.__safe_dict_get(view, 'checked', default=False)
            selected = self.__safe_dict_get(view, 'selected', default=False)
            content_description = self.__safe_dict_get(view, 'content_description', default='')
            view_text = self.__safe_dict_get(view, 'text', default='')
            view_class = self.__safe_dict_get(view, 'class').split('.')[-1]
            if not content_description and not view_text and not scrollable:  # actionable?
                continue
            
            # text = self._merge_text(view_text, content_description)
            # view_status = ''
            if editable:
                # view_status += 'editable '
                view_desc = input_frame.replace('@', str(len(view_descs))).replace('#', view_text)
                if content_description:
                    view_desc = view_desc.replace('&', content_description)
                else:
                    view_desc = view_desc.replace(" class='&'", "")
                view_descs.append(view_desc)
                available_actions.append(SetTextEvent(view=view, text='HelloWorld'))
            elif (clickable or checkable or long_clickable):
                
                view_desc = btn_frame.replace('@', str(len(view_descs))).replace('#', view_text).replace('$', str(checked or selected))
                # import pdb;pdb.set_trace()
                if content_description:
                    view_desc = view_desc.replace('&', content_description)
                else:
                    view_desc = view_desc.replace(" class='&'", "")
                view_descs.append(view_desc)

                available_actions.append(TouchEvent(view=view))
            elif scrollable:
                view_descs.append(scroll_up_frame.replace('@', str(len(view_descs))))#.replace('&', view_class).replace('#', text))
                available_actions.append(ScrollEvent(view=view, direction='UP'))
                view_descs.append(scroll_down_frame.replace('@', str(len(view_descs))))#.replace('&', view_class).replace('#', text))
                available_actions.append(ScrollEvent(view=view, direction='DOWN'))
            else:
                view_desc = text_frame.replace('@', str(len(view_descs))).replace('#', view_text)

                if content_description:
                    view_desc = view_desc.replace('&', content_description)
                else:
                    view_desc = view_desc.replace(" class='&'", "")
                view_descs.append(view_desc)

                available_actions.append(TouchEvent(view=view))
        view_descs.append(f"<button id={len(view_descs)} class='ImageButton'>go back</button>")
        available_actions.append(KeyEvent(name='BACK'))
        # state_desc = 'The current state has the following UI elements: \n' #views and corresponding actions, with action id in parentheses:\n '
        state_desc = prefix #'Given a screen, an instruction, predict the id of the UI element to perform the insturction. The screen has the following UI elements: \n'
        # state_desc = 'You can perform actions on a contacts app, the current state of which has the following UI views and corresponding actions, with action id in parentheses:\n'
        state_desc += '\n '.join(view_descs)
        
        views_without_id = self._remove_view_ids(view_descs)

        return state_desc, available_actions, views_without_id

    # def _build_view_tree(self):
    #     # import networkx as nx
        
    #     view_tree = Tree()#nx.DiGraph()
    #     node_desc = 0
    #     view_tree.create_node(node_desc, 0)
    #     for view_id in range(1, len(self.views)):
    #         view = self.views[view_id]
    #         parentid = view['parent']
    #         # node_desc = view['text'] + '[|*]+' + view['content']
    #         view_tree.create_node(view_id, view_id, parent=parentid)
    #     view_tree.show()
    #     for node in view_tree.expand_tree(mode=Tree.WIDTH, sorting=False):
    #         if self.views[node]['clickable']:
    #             print(node, self.views[node]['text'], self.views[node]['content_description'])            
    #     # import pdb;pdb.set_trace()
    #     return view_tree
    def _build_view_graph(self):        
        view_graph = nx.DiGraph()
        for view_id in range(1, len(self.views)):
            view = self.views[view_id]
            parentid = view['parent']
            view_graph.add_edge(parentid, view_id)
        # self.visualize_graph(view_graph)
        return view_graph

    def visualize_graph(self, graph):
        import matplotlib.pyplot as plt
        nx.draw(graph, with_labels=True, font_weight='bold')
        plt.show()
    
    def _adjust_view_clickability(self):
        '''make the view unclickable if it has clickable successors'''
        for view_id in range(1, len(self.views)):
            if self.__safe_dict_get(self.views[view_id], 'clickable', default=False):
                successors = self._extract_all_children(view_id)
                # print('origin:', view_id, 'succs: ', successors)
                for successor in successors:
                    if successor != view_id and self.__safe_dict_get(self.views[successor], 'clickable', False):
                        # print(self.views[view_id], 'disabled, because of ', self.views[successor])
                        self.views[view_id]['clickable'] = False
                        # print('origin:', view_id, 'because of:', successor, 'disabled')
                        break
            if self.__safe_dict_get(self.views[view_id], 'checkable', default=False):
                successors = self._extract_all_children(view_id)
                for successor in successors:
                    if successor != view_id and self.__safe_dict_get(self.views[successor], 'checkable', False):
                        self.views[view_id]['checkable'] = False
                        break
    
    def _get_ancestor_id(self, view, key, default=None):
        if self.__safe_dict_get(view, key=key, default=False):
            return view['temp_id']
        all_views = [view] + [self.views[i] for i in self.get_all_ancestors(view)]
        for v in all_views:
            value = self.__safe_dict_get(v, key)
            if value:
                return v['temp_id']
        return default
    
    def _extract_all_children(self, id):
        successors = []
        successors_of_view = nx.dfs_successors(self.view_graph, source=id, depth_limit=100)
        # print(successors_of_view)
        for k, v in successors_of_view.items():
            for successor_id in v:
                if successor_id not in successors and successor_id != id:
                    successors.append(successor_id)
        return successors
        # if len(self.viewtree.children(id)) == 0:
        #     return
        # else:
        #     for ch_ele in self.viewtree.children(id):
        #         childrenlist.append(ch_ele)
        #         self._extract_all_children(ch_ele, childrenlist)
        # return childrenlist
    def _merge_textv2(self, children_ids, remove_time_and_ip=False, important_view_ids=[]):
        texts, content_descriptions = [], []
        for childid in children_ids:

            if not self.__safe_dict_get(self.views[childid], 'visible') or \
                self.__safe_dict_get(self.views[childid], 'resource_id') in \
               ['android:id/navigationBarBackground',
                'android:id/statusBarBackground']:
                # if the successor is not visible, then ignore it!
                continue          

            text = self.__safe_dict_get(self.views[childid], 'text', default='')
            if len(text) > 50:
                text = text[:50]

            if remove_time_and_ip:
                text = self._remove_ip_and_date(text)

            if text != '':
                # text = text + '  {'+ str(childid)+ '}'
                texts.append(text)
                important_view_ids.append([text, childid])

            content_description = self.__safe_dict_get(self.views[childid], 'content_description', default='')
            if len(content_description) > 50:
                content_description = content_description[:50]

            if remove_time_and_ip:
                content_description = self._remove_ip_and_date(content_description)

            if content_description != '':
                # content_description = content_description + '  {'+ str(childid)+ '}'
                important_view_ids.append([content_description, childid])
                content_descriptions.append(content_description)

        merged_text = '<br>'.join(texts) if len(texts) > 0 else ''
        merged_desc = '<br>'.join(content_descriptions) if len(content_descriptions) > 0 else ''
        return merged_text, merged_desc, important_view_ids
    
    def _get_children_checked(self, children_ids):
        for childid in children_ids:
            if self.__safe_dict_get(self.views[childid], 'checked', default=False):
                return True
        return False
    
    def _get_children_checkable(self, children_ids):
        for childid in children_ids:
            if self.__safe_dict_get(self.views[childid], 'checkable', default=False):
                return True
        return False
    
    def _has_clickable_children(self, id):
        children = self._extract_all_children(id)
        # children = 
        for child_view_id in children:
            clickable = self.__safe_dict_get(self.views[child_view_id], 'clickable', default=False)
            checkable = self.__safe_dict_get(self.views[child_view_id], 'checkable', default=False) 
            if clickable or checkable:
                return True
        return False
    
    # def _self_not_clickable_with_clickable_relatives(self, id):
    #     if self.__safe_dict_get(self.views[id], key='clickable', default=False) or self.__safe_dict_get(self.views[id], key='checkable', default=False):
    #         return False

    # def _adjust_view_clickability(self):
    #     '''make the view unclickable if it has clickable successors'''
    #     for view_id in range(1, len(self.views)):
    #         if self.__safe_dict_get(self.views[view_id], 'clickable', default=False):
    #             successors = self._extract_all_children(view_id)
    #             # print('origin:', view_id, 'succs: ', successors)
    #             for successor in successors:
    #                 if successor != view_id and self.__safe_dict_get(self.views[successor], 'clickable', False):
    #                     # print(self.views[view_id], 'disabled, because of ', self.views[successor])
    #                     self.views[view_id]['clickable'] = False
    #                     print('origin:', view_id, 'because of:', successor, 'disabled')
    #                     break
    #         if self.__safe_dict_get(self.views[view_id], 'checkable', default=False):
    #             successors = self._extract_all_children(view_id)
    #             for successor in successors:
    #                 if successor != view_id and self.__safe_dict_get(self.views[successor], 'checkable', False):
    #                     self.views[view_id]['checkable'] = False
    #                     break

    
    def get_described_actions(self, prefix='', remove_time_and_ip=False,
                                merge_buttons =True, add_edit_box = True, add_check_box = True, add_pure_text = True):
        """
        Get a text description of current state
        """
        enabled_view_ids = []
        for view_dict in self.views:
            # exclude navigation bar if exists
            if self.__safe_dict_get(view_dict, 'visible') and \
                self.__safe_dict_get(view_dict, 'resource_id') not in \
               ['android:id/navigationBarBackground',
                'android:id/statusBarBackground']:
                enabled_view_ids.append(view_dict['temp_id'])
        
        text_frame = "<p id=@ text='&'>#</p>"
        btn_frame = "<button id=@ text='&'>#</button>"
        checkbox_frame = "<checkbox id=@ checked=$ text='&'>#</checkbox>"
        input_frame = "<input id=@ text='&'>#</input>"
        scroll_down_frame = "<div id=@ class='scroller'>scroll down</div>"
        scroll_up_frame = "<div id=@ class='scroller'>scroll up</div>"

        view_descs = []
        available_actions = []
        removed_view_ids = []

        important_view_ids = []

        for view_id in enabled_view_ids:
            if view_id in removed_view_ids:
                continue
            # print(view_id)
            view = self.views[view_id]
            clickable = self._get_self_ancestors_property(view, 'clickable')
            scrollable = self.__safe_dict_get(view, 'scrollable')
            checkable = self._get_self_ancestors_property(view, 'checkable')
            long_clickable = self._get_self_ancestors_property(view, 'long_clickable')
            editable = self.__safe_dict_get(view, 'editable')
            actionable = clickable or scrollable or checkable or long_clickable or editable
            checked = self.__safe_dict_get(view, 'checked', default=False)
            selected = self.__safe_dict_get(view, 'selected', default=False)
            content_description = self.__safe_dict_get(view, 'content_description', default='')
            view_text = self.__safe_dict_get(view, 'text', default='')
            view_class = self.__safe_dict_get(view, 'class').split('.')[-1]
            if not content_description and not view_text and not scrollable:  # actionable?
                continue

            # text = self._merge_text(view_text, content_description)
            # view_status = ''
            if editable:
                # view_status += 'editable '
                view_desc = input_frame.replace('@', str(len(view_descs))).replace('#', view_text)
                if content_description:
                    view_desc = view_desc.replace('&', content_description)
                else:
                    view_desc = view_desc.replace(" text='&'", "")
                # view_desc = view_desc.replace('*&*', str(view_id))
                view_descs.append(view_desc)
                available_actions.append(SetTextEvent(view=view, text='HelloWorld'))
                important_view_ids.append([content_description + view_text,view_id])

            elif checkable:
                view_desc = checkbox_frame.replace('@', str(len(view_descs))).replace('#', view_text).replace('$',
                                                                                                              str(checked or selected))
                if content_description:
                    view_desc = view_desc.replace('&', content_description)
                else:
                    view_desc = view_desc.replace(" text='&'", "")
                view_descs.append(view_desc)
                if add_check_box:
                    available_actions.append(TouchEvent(view=view))
                else:
                    available_actions.append(None)
                # view_dict_list.append(
                #     {'id': len(view_descs) - 1, 'text': view_text, 'content_description': content_description,
                #      'checked': checked or selected, 'type': 'checkbox'})
            elif clickable:  # or long_clickable
                if merge_buttons:
                    # below is to merge buttons, led to bugs
                    clickable_ancestor_id = self._get_ancestor_id(view=view, key='clickable')
                    if not clickable_ancestor_id:
                        clickable_ancestor_id = self._get_ancestor_id(view=view, key='checkable')
                    # if not clickable_ancestor_id:
                    #     clickable_ancestor_id = self._get_ancestor_id(view=view, key='long_clickable')
                    clickable_children_ids = self._extract_all_children(id=clickable_ancestor_id)

                    if view_id not in clickable_children_ids:
                        clickable_children_ids.append(view_id)

                    view_text, content_description, important_view_ids = self._merge_textv2(clickable_children_ids,
                                                                                            remove_time_and_ip,
                                                                                            important_view_ids)
                    checked = self._get_children_checked(clickable_children_ids)
                    # end of merging buttons
                if not view_text and not content_description:
                    continue
                view_desc = btn_frame.replace('@', str(len(view_descs))).replace('#', view_text)
                if content_description:
                    view_desc = view_desc.replace('&', content_description)
                else:
                    view_desc = view_desc.replace(" text='&'", "")
                view_descs.append(view_desc)

                available_actions.append(TouchEvent(view=view))
                # if view_id == 111:
                #     pdb.set_trace()
                if merge_buttons:
                    for clickable_child in clickable_children_ids:
                        if clickable_child in enabled_view_ids and clickable_child != view_id:
                            removed_view_ids.append(clickable_child)

                # view_dict_list.append(
                #     {'id': len(view_descs) - 1, 'text': view_text, 'content_description': content_description,
                #      'type': 'button'})


            elif scrollable:
                # print(view_id, 'continued')
                continue
                # view_descs.append(scroll_up_frame.replace('@', str(len(view_descs))))#.replace('&', view_class).replace('#', text))
                # available_actions.append(ScrollEvent(view=view, direction='UP'))
                # view_descs.append(scroll_down_frame.replace('@', str(len(view_descs))))#.replace('&', view_class).replace('#', text))
                # available_actions.append(ScrollEvent(view=view, direction='DOWN'))
            else:
                
                if remove_time_and_ip:
                    view_text = self._remove_ip_and_date(view_text)
                    content_description = self._remove_ip_and_date(content_description)

                view_desc = text_frame.replace('@', str(len(view_descs))).replace('#', view_text)

                if content_description:
                    view_desc = view_desc.replace('&', content_description)
                else:
                    view_desc = view_desc.replace(" text='&'", "")
                view_descs.append(view_desc)

                important_view_ids.append([content_description + view_text,view_id])

                available_actions.append(TouchEvent(view=view))
        view_descs.append(f"<button id={len(view_descs)}>go back</button>")
        available_actions.append(KeyEvent(name='BACK'))
        # state_desc = 'The current state has the following UI elements: \n' #views and corresponding actions, with action id in parentheses:\n '
        state_desc = prefix #'Given a screen, an instruction, predict the id of the UI element to perform the insturction. The screen has the following UI elements: \n'
        # state_desc = 'You can perform actions on a contacts app, the current state of which has the following UI views and corresponding actions, with action id in parentheses:\n'
        state_desc += '\n'.join(view_descs)
        
        views_without_id = self._remove_view_ids(view_descs)
        # print(views_without_id)
        return state_desc, available_actions, views_without_id, important_view_ids
    
    def get_view_desc(self, view):
        content_description = self.__safe_dict_get(view, 'content_description', default='')
        view_text = self.__safe_dict_get(view, 'text', default='')
        scrollable = self.__safe_dict_get(view, 'scrollable')
        clickable = self._get_self_ancestors_property(view, 'clickable')
        checkable = self._get_self_ancestors_property(view, 'checkable')
        long_clickable = self._get_self_ancestors_property(view, 'long_clickable')
        editable = self.__safe_dict_get(view, 'editable')
        view_class = self.__safe_dict_get(view, 'class').split('.')[-1]
        text = self._merge_text(view_text, content_description)
        checked = self.__safe_dict_get(view, 'checked', default=False)
        selected = self.__safe_dict_get(view, 'selected', default=False)

        # view_desc = f'view'
        # btn_frame = "<button id=@ checked=$ class='&' label='~'>#</button>"
        # input_frame = "<input id=@ class='&' >#</input>"
        if editable:
                # view_status += 'editable '
            view_desc = f"<input class='&'>#</input>"#.replace('&', view_class)#.replace('#', text)
            if view_text:
                view_desc = view_desc.replace('#', view_text)
            else:
                view_desc = view_desc.replace('#', '')
            if content_description:
                view_desc = view_desc.replace('&', content_description)
            else:
                view_desc = view_desc.replace(" class='&'", "")
            # available_actions.append(SetTextEvent(view=view, text='HelloWorld'))
        elif (clickable or checkable or long_clickable):

            view_id = view['temp_id']

            clickable_ancestor_id = self._get_ancestor_id(view=view, key='clickable')
            if not clickable_ancestor_id:
                clickable_ancestor_id = self._get_ancestor_id(view=view, key='checkable')
            if not clickable_ancestor_id:
                clickable_ancestor_id = self._get_ancestor_id(view=view, key='long_clickable')
            clickable_children_ids = self._extract_all_children(id=clickable_ancestor_id)

            if view_id not in clickable_children_ids:
                clickable_children_ids.append(view_id)
                
            view_text, content_description, important_view_ids = self._merge_textv2(clickable_children_ids, False, [])
            checked = self._get_children_checked(clickable_children_ids)
                # print(view_id, clickable_ancestor_id, clickable_children_ids, view_text, content_description)

                # view_desc = btn_frame.replace('@', str(len(view_descs))).replace('#', view_text).replace('$', str(checked or selected))

            view_desc = f"<button checked=$ class='&'>#</button>".replace('$', str(checked or selected))
            if view_text:
                view_desc = view_desc.replace('#', view_text)
            else:
                view_desc = view_desc.replace('#', '')
            if content_description:
                view_desc = view_desc.replace('&', content_description)
            else:
                view_desc = view_desc.replace(" class='&'", "")

            # available_actions.append(TouchEvent(view=view))
        elif scrollable:
            view_desc = f"<div class='scroller'>scroll the screen</div>"
        else:
            view_desc = f"<p class='&'>#</p>"
            if view_text:
                view_desc = view_desc.replace('#', view_text)
            else:
                view_desc = view_desc.replace('#', '')
            if content_description:
                view_desc = view_desc.replace('&', content_description)
            else:
                view_desc = view_desc.replace(" class='&'", "")
        return view_desc
    
    def get_action_desc(self, action):
        desc = action.event_type
        if isinstance(action, KeyEvent):
            view_desc = "<button class='ImageButton'>go back</button>"
            desc = '- TapOn: ' + view_desc
            # desc = view_desc + '.click();'
            # desc = f'- go {action.name.lower()}'
        if isinstance(action, UIEvent):
            view_desc = self.get_view_desc(action.view)
            # action_name = action.event_type
            if isinstance(action, LongTouchEvent):
                # desc = view_desc + '.longclick();'
                desc = '- LongTapOn: ' + view_desc
            elif isinstance(action, SetTextEvent):
                # action_name = f'enter "{action.text}" into'
                # desc = view_desc + f'.settext{action.text}'
                desc = '- TapOn: ' + view_desc  + ' InputText: ' + action.text
            elif isinstance(action, ScrollEvent):
                # action_name = f'scroll {action.direction.lower()}'
                # desc = view_desc + f'.scroll{action.direction.lower()}'
                desc = f'- Scroll{action.direction.lower()}: ' + view_desc
            else:
                # desc = view_desc + '.click()'
                desc = '- TapOn: ' + view_desc
            # desc = f'- {action_name} {self.get_view_desc(action.view)}'
        return desc
    
    def get_action_descv2(self, action, view_desc):
        desc = action.event_type
        if isinstance(action, KeyEvent):
            desc = '- TapOn: ' + view_desc
        if isinstance(action, UIEvent):
            if isinstance(action, LongTouchEvent):
                desc = '- LongTapOn: ' + view_desc
            elif isinstance(action, SetTextEvent):
                desc = '- TapOn: ' + view_desc  + ' InputText: ' + action.text
            elif isinstance(action, ScrollEvent):
                desc = f'- Scroll{action.direction.lower()}: ' + view_desc
            else:
                desc = '- TapOn: ' + view_desc
        return desc
    
    def view_scrollable(self, view_dict):
        visible = False
        # exclude navigation bar if exists
        if self.__safe_dict_get(view_dict, 'visible') and \
            self.__safe_dict_get(view_dict, 'resource_id') not in \
            ['android:id/navigationBarBackground',
            'android:id/statusBarBackground']:
            visible = True
        if not visible:
            return False

        clickable = self._get_self_ancestors_property(view_dict, 'clickable')
        scrollable = self.__safe_dict_get(view_dict, 'scrollable')
        checkable = self._get_self_ancestors_property(view_dict, 'checkable')
        long_clickable = self._get_self_ancestors_property(view_dict, 'long_clickable')
        editable = self.__safe_dict_get(view_dict, 'editable')
        # actionable = clickable or scrollable or checkable or long_clickable or editable
        # checked = self.__safe_dict_get(view_dict, 'checked')
        # selected = self.__safe_dict_get(view_dict, 'selected')
        # content_description = self.__safe_dict_get(view_dict, 'content_description', default='')
        # view_text = self.__safe_dict_get(view_dict, 'text', default='')
        # view_class = self.__safe_dict_get(view_dict, 'class').split('.')[-1]

        if editable or clickable or checkable or long_clickable:
            return False
        elif scrollable:
            return True
        else:
            return False
    
    def _remove_ip_and_date(self, string, remove_candidates=None):
        if not string:
            return string
        import re
        if not remove_candidates:
            remove_candidates = ['hr', 'min', 'sec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'\
                                 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December',
                                 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
                                 'Sun', 'Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat']  # '[0-9]+',
        for remove_candidate in remove_candidates:
            string = re.sub(remove_candidate, '', string)
        if ':' in string or '::' in string or '%' in string:  # ip address, hour, storage usage
            string = ''
        # if '::' in string:  # ip address
        #     string = ''
        return string
    
        
    def get_number_free_screen(self, prefix=''):
        """
        Get a text description of current state
        """

        enabled_view_ids = []
        for view_dict in self.views:
            # exclude navigation bar if exists
            if self.__safe_dict_get(view_dict, 'visible') and \
                self.__safe_dict_get(view_dict, 'resource_id') not in \
               ['android:id/navigationBarBackground',
                'android:id/statusBarBackground']:
                enabled_view_ids.append(view_dict['temp_id'])
        
        text_frame = "<p id=@ class='&'>#</p>"
        btn_frame = "<button id=@ class='&' checked=$>#</button>"
        input_frame = "<input id=@ class='&' >#</input>"
        scroll_down_frame = "<div id=@ class='scroller'>scroll down</div>"
        scroll_up_frame = "<div id=@ class='scroller'>scroll up</div>"

        view_descs = []
        available_actions = []
        for view_id in enabled_view_ids:
            view = self.views[view_id]
            clickable = self._get_self_ancestors_property(view, 'clickable')
            scrollable = self.__safe_dict_get(view, 'scrollable')
            checkable = self._get_self_ancestors_property(view, 'checkable')
            long_clickable = self._get_self_ancestors_property(view, 'long_clickable')
            editable = self.__safe_dict_get(view, 'editable')
            actionable = clickable or scrollable or checkable or long_clickable or editable
            checked = self.__safe_dict_get(view, 'checked', default=False)
            selected = self.__safe_dict_get(view, 'selected', default=False)
            content_description = self.__safe_dict_get(view, 'content_description', default='')
            view_text = self.__safe_dict_get(view, 'text', default='')
            view_class = self.__safe_dict_get(view, 'class').split('.')[-1]
            if not content_description and not view_text and not scrollable:  # actionable?
                continue

            content_description = self._remove_date_and_date(content_description)
            view_text = self._remove_date_and_date(view_text)
            # text = self._merge_text(view_text, content_description)
            # view_status = ''
            if editable:
                # view_status += 'editable '
                view_desc = input_frame.replace('@', str(len(view_descs))).replace('#', view_text)
                if content_description:
                    view_desc = view_desc.replace('&', content_description)
                else:
                    view_desc = view_desc.replace(" class='&'", "")
                view_descs.append(view_desc)
                available_actions.append(SetTextEvent(view=view, text='HelloWorld'))
            elif (clickable or checkable or long_clickable):
                
                view_desc = btn_frame.replace('@', str(len(view_descs))).replace('#', view_text).replace('$', str(checked or selected))
                # import pdb;pdb.set_trace()
                if content_description:
                    view_desc = view_desc.replace('&', content_description)
                else:
                    view_desc = view_desc.replace(" class='&'", "")
                view_descs.append(view_desc)

                available_actions.append(TouchEvent(view=view))
            elif scrollable:
                view_descs.append(scroll_up_frame.replace('@', str(len(view_descs))))#.replace('&', view_class).replace('#', text))
                available_actions.append(ScrollEvent(view=view, direction='UP'))
                view_descs.append(scroll_down_frame.replace('@', str(len(view_descs))))#.replace('&', view_class).replace('#', text))
                available_actions.append(ScrollEvent(view=view, direction='DOWN'))
            else:
                view_desc = text_frame.replace('@', str(len(view_descs))).replace('#', view_text)

                if content_description:
                    view_desc = view_desc.replace('&', content_description)
                else:
                    view_desc = view_desc.replace(" class='&'", "")
                view_descs.append(view_desc)

                available_actions.append(TouchEvent(view=view))
        view_descs.append(f"<button id={len(view_descs)} class='ImageButton'>go back</button>")
        available_actions.append(KeyEvent(name='BACK'))
        # state_desc = 'The current state has the following UI elements: \n' #views and corresponding actions, with action id in parentheses:\n '
        state_desc = prefix #'Given a screen, an instruction, predict the id of the UI element to perform the insturction. The screen has the following UI elements: \n'
        # state_desc = 'You can perform actions on a contacts app, the current state of which has the following UI views and corresponding actions, with action id in parentheses:\n'
        state_desc += '\n '.join(view_descs)
        
        views_without_id = self._remove_view_ids(view_descs)

        return state_desc#, available_actions, views_without_id
    
    def get_scrollable_views(self):
        scrollable_views = []
        enabled_view_ids = []
        for view_dict in self.views:
            # exclude navigation bar if exists
            if self.__safe_dict_get(view_dict, 'visible') and \
                self.__safe_dict_get(view_dict, 'resource_id') not in \
               ['android:id/navigationBarBackground',
                'android:id/statusBarBackground']:
                enabled_view_ids.append(view_dict['temp_id'])
        for view_id in enabled_view_ids:
            view = self.views[view_id]
            scrollable = self.__safe_dict_get(view, 'scrollable')

            clickable = self._get_self_ancestors_property(view, 'clickable')
            scrollable = self.__safe_dict_get(view, 'scrollable')
            checkable = self._get_self_ancestors_property(view, 'checkable')
            long_clickable = self._get_self_ancestors_property(view, 'long_clickable')
            editable = self.__safe_dict_get(view, 'editable')

            if scrollable and not clickable and not checkable and not long_clickable and not editable:
                scrollable_views.append(view)
        return scrollable_views