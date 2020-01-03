# ML-Support-Vector-Machine-Examples
This a an example of a machine learning model using support vector machine algorithm. 
<div class="cell text_cell rendered selected" tabindex="2">
<div class="inner_cell">
<div class="text_cell_render rendered_html" tabindex="-1">
<h2 id="Support-Vector-Machine-Tutorial-Using-Python-Sklearn" style="color: blue;" align="center">Support Vector Machine Tutorial Using Python Sklearn</h2>
</div>
</div>
</div>
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 327.8px; margin-bottom: 0px; border-right-width: 30px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-cursors" style="visibility: hidden;">&nbsp;</div>
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-keyword">import</span> <span class="cm-variable">pandas</span> <span class="cm-keyword">as</span> <span class="cm-variable">pd</span></pre>
<pre class=" CodeMirror-line "><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">datasets</span> <span class="cm-keyword">import</span> <span class="cm-variable">load_iris</span></pre>
<pre class=" CodeMirror-line "><span class="cm-variable">iris</span> <span class="cm-operator">=</span> <span class="cm-variable">load_iris</span><span class=" CodeMirror-matchingbracket">(</span><span class=" CodeMirror-matchingbracket">)</span></pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 62px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="cell text_cell rendered unselected" tabindex="2">
<div class="inner_cell">
<div class="text_cell_render rendered_html" tabindex="-1">
<p><img src="https://github.com/Way4ward17/ML-Support-Vector-Machine-Examples/blob/master/sepal.png" alt="" width="300" height="300" /></p>
</div>
</div>
</div>
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 159.8px; margin-bottom: 0px; border-right-width: 30px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">iris</span>.<span class="cm-property">feature_names</span></pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 28px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_area">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_text output_result">
<pre>['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)']</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 151.4px; margin-bottom: 0px; border-right-width: 30px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">iris</span>.<span class="cm-property">target_names</span></pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 28px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_area">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_text output_result">
<pre>array(['setosa', 'versicolor', 'virginica'], dtype='&lt;U10')</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 470.6px; margin-bottom: 0px; border-right-width: 30px; min-height: 45px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">df</span> <span class="cm-operator">=</span> <span class="cm-variable">pd</span>.<span class="cm-property">DataFrame</span>(<span class="cm-variable">iris</span>.<span class="cm-property">data</span>,<span class="cm-variable">columns</span><span class="cm-operator">=</span><span class="cm-variable">iris</span>.<span class="cm-property">feature_names</span>)</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">df</span>.<span class="cm-property">head</span>()</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 45px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="output_area">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_html rendered_html output_result">
<div>
<table class="dataframe" border="1">
<thead>
<tr style="text-align: right;">
<th>&nbsp;</th>
<th>sepal length (cm)</th>
<th>sepal width (cm)</th>
<th>petal length (cm)</th>
<th>petal width (cm)</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>5.1</td>
<td>3.5</td>
<td>1.4</td>
<td>0.2</td>
</tr>
<tr>
<td>1</td>
<td>4.9</td>
<td>3.0</td>
<td>1.4</td>
<td>0.2</td>
</tr>
<tr>
<td>2</td>
<td>4.7</td>
<td>3.2</td>
<td>1.3</td>
<td>0.2</td>
</tr>
<tr>
<td>3</td>
<td>4.6</td>
<td>3.1</td>
<td>1.5</td>
<td>0.2</td>
</tr>
<tr>
<td>4</td>
<td>5.0</td>
<td>3.6</td>
<td>1.4</td>
<td>0.2</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="prompt_container">&nbsp;</div>
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 227px; margin-bottom: 0px; border-right-width: 30px; min-height: 45px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">df</span>[<span class="cm-string">'target'</span>] <span class="cm-operator">=</span> <span class="cm-variable">iris</span>.<span class="cm-property">target</span></pre>
<pre class=" CodeMirror-line "><span class="cm-variable">df</span>.<span class="cm-property">head</span>()</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 45px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="output_area">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_html rendered_html output_result">
<div>
<table class="dataframe" border="1">
<thead>
<tr style="text-align: right;">
<th>&nbsp;</th>
<th>sepal length (cm)</th>
<th>sepal width (cm)</th>
<th>petal length (cm)</th>
<th>petal width (cm)</th>
<th>target</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>5.1</td>
<td>3.5</td>
<td>1.4</td>
<td>0.2</td>
<td>0</td>
</tr>
<tr>
<td>1</td>
<td>4.9</td>
<td>3.0</td>
<td>1.4</td>
<td>0.2</td>
<td>0</td>
</tr>
<tr>
<td>2</td>
<td>4.7</td>
<td>3.2</td>
<td>1.3</td>
<td>0.2</td>
<td>0</td>
</tr>
<tr>
<td>3</td>
<td>4.6</td>
<td>3.1</td>
<td>1.5</td>
<td>0.2</td>
<td>0</td>
</tr>
<tr>
<td>4</td>
<td>5.0</td>
<td>3.6</td>
<td>1.4</td>
<td>0.2</td>
<td>0</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="prompt_container">&nbsp;</div>
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.60001px; left: 5.60001px;">&nbsp;</div>
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 201.8px; margin-bottom: 0px; border-right-width: 30px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">df</span>[<span class="cm-variable">df</span>.<span class="cm-property">target</span><span class="cm-operator">==</span><span class="cm-number">1</span>].<span class="cm-property">head</span>()</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 28px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="output_area">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_html rendered_html output_result">
<div>
<table class="dataframe" border="1">
<thead>
<tr style="text-align: right;">
<th>&nbsp;</th>
<th>sepal length (cm)</th>
<th>sepal width (cm)</th>
<th>petal length (cm)</th>
<th>petal width (cm)</th>
<th>target</th>
</tr>
</thead>
<tbody>
<tr>
<td>50</td>
<td>7.0</td>
<td>3.2</td>
<td>4.7</td>
<td>1.4</td>
<td>1</td>
</tr>
<tr>
<td>51</td>
<td>6.4</td>
<td>3.2</td>
<td>4.5</td>
<td>1.5</td>
<td>1</td>
</tr>
<tr>
<td>52</td>
<td>6.9</td>
<td>3.1</td>
<td>4.9</td>
<td>1.5</td>
<td>1</td>
</tr>
<tr>
<td>53</td>
<td>5.5</td>
<td>2.3</td>
<td>4.0</td>
<td>1.3</td>
<td>1</td>
</tr>
<tr>
<td>54</td>
<td>6.5</td>
<td>2.8</td>
<td>4.6</td>
<td>1.5</td>
<td>1</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="prompt_container">&nbsp;</div>
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 201.8px; margin-bottom: 0px; border-right-width: 30px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">df</span>[<span class="cm-variable">df</span>.<span class="cm-property">target</span><span class="cm-operator">==</span><span class="cm-number">2</span>].<span class="cm-property">head</span>()</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 28px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="output_area">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_html rendered_html output_result">
<div>
<table class="dataframe" border="1">
<thead>
<tr style="text-align: right;">
<th>&nbsp;</th>
<th>sepal length (cm)</th>
<th>sepal width (cm)</th>
<th>petal length (cm)</th>
<th>petal width (cm)</th>
<th>target</th>
</tr>
</thead>
<tbody>
<tr>
<td>100</td>
<td>6.3</td>
<td>3.3</td>
<td>6.0</td>
<td>2.5</td>
<td>2</td>
</tr>
<tr>
<td>101</td>
<td>5.8</td>
<td>2.7</td>
<td>5.1</td>
<td>1.9</td>
<td>2</td>
</tr>
<tr>
<td>102</td>
<td>7.1</td>
<td>3.0</td>
<td>5.9</td>
<td>2.1</td>
<td>2</td>
</tr>
<tr>
<td>103</td>
<td>6.3</td>
<td>2.9</td>
<td>5.6</td>
<td>1.8</td>
<td>2</td>
</tr>
<tr>
<td>104</td>
<td>6.5</td>
<td>3.0</td>
<td>5.8</td>
<td>2.2</td>
<td>2</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="prompt_container">&nbsp;</div>
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.59998px; left: 5.60001px;">&nbsp;</div>
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 563px; margin-bottom: 0px; border-right-width: 30px; min-height: 45px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">df</span>[<span class="cm-string">'flower_name'</span>] <span class="cm-operator">=</span><span class="cm-variable">df</span>.<span class="cm-property">target</span>.<span class="cm-property">apply</span>(<span class="cm-keyword">lambda</span> <span class="cm-variable">x</span>: <span class="cm-variable">iris</span>.<span class="cm-property">target_names</span>[<span class="cm-variable">x</span>])</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">df</span>.<span class="cm-property">head</span>()</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 45px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="output_area">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_html rendered_html output_result">
<div>
<table class="dataframe" border="1">
<thead>
<tr style="text-align: right;">
<th>&nbsp;</th>
<th>sepal length (cm)</th>
<th>sepal width (cm)</th>
<th>petal length (cm)</th>
<th>petal width (cm)</th>
<th>target</th>
<th>flower_name</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>5.1</td>
<td>3.5</td>
<td>1.4</td>
<td>0.2</td>
<td>0</td>
<td>setosa</td>
</tr>
<tr>
<td>1</td>
<td>4.9</td>
<td>3.0</td>
<td>1.4</td>
<td>0.2</td>
<td>0</td>
<td>setosa</td>
</tr>
<tr>
<td>2</td>
<td>4.7</td>
<td>3.2</td>
<td>1.3</td>
<td>0.2</td>
<td>0</td>
<td>setosa</td>
</tr>
<tr>
<td>3</td>
<td>4.6</td>
<td>3.1</td>
<td>1.5</td>
<td>0.2</td>
<td>0</td>
<td>setosa</td>
</tr>
<tr>
<td>4</td>
<td>5.0</td>
<td>3.6</td>
<td>1.4</td>
<td>0.2</td>
<td>0</td>
<td>setosa</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="prompt_container">&nbsp;</div>
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 84.2px; margin-bottom: 0px; border-right-width: 30px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">df</span>[<span class="cm-number">45</span>:<span class="cm-number">55</span>]</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 28px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_area">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_html rendered_html output_result">
<div>
<table class="dataframe" border="1">
<thead>
<tr style="text-align: right;">
<th>&nbsp;</th>
<th>sepal length (cm)</th>
<th>sepal width (cm)</th>
<th>petal length (cm)</th>
<th>petal width (cm)</th>
<th>target</th>
<th>flower_name</th>
</tr>
</thead>
<tbody>
<tr>
<td>45</td>
<td>4.8</td>
<td>3.0</td>
<td>1.4</td>
<td>0.3</td>
<td>0</td>
<td>setosa</td>
</tr>
<tr>
<td>46</td>
<td>5.1</td>
<td>3.8</td>
<td>1.6</td>
<td>0.2</td>
<td>0</td>
<td>setosa</td>
</tr>
<tr>
<td>47</td>
<td>4.6</td>
<td>3.2</td>
<td>1.4</td>
<td>0.2</td>
<td>0</td>
<td>setosa</td>
</tr>
<tr>
<td>48</td>
<td>5.3</td>
<td>3.7</td>
<td>1.5</td>
<td>0.2</td>
<td>0</td>
<td>setosa</td>
</tr>
<tr>
<td>49</td>
<td>5.0</td>
<td>3.3</td>
<td>1.4</td>
<td>0.2</td>
<td>0</td>
<td>setosa</td>
</tr>
<tr>
<td>50</td>
<td>7.0</td>
<td>3.2</td>
<td>4.7</td>
<td>1.4</td>
<td>1</td>
<td>versicolor</td>
</tr>
<tr>
<td>51</td>
<td>6.4</td>
<td>3.2</td>
<td>4.5</td>
<td>1.5</td>
<td>1</td>
<td>versicolor</td>
</tr>
<tr>
<td>52</td>
<td>6.9</td>
<td>3.1</td>
<td>4.9</td>
<td>1.5</td>
<td>1</td>
<td>versicolor</td>
</tr>
<tr>
<td>53</td>
<td>5.5</td>
<td>2.3</td>
<td>4.0</td>
<td>1.3</td>
<td>1</td>
<td>versicolor</td>
</tr>
<tr>
<td>54</td>
<td>6.5</td>
<td>2.8</td>
<td>4.6</td>
<td>1.5</td>
<td>1</td>
<td>versicolor</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="prompt_container">&nbsp;</div>
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.59998px; left: 5.60001px;">&nbsp;</div>
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 143px; margin-bottom: 0px; border-right-width: 30px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">df0</span> <span class="cm-operator">=</span> <span class="cm-variable">df</span>[:<span class="cm-number">50</span>]</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">df1</span> <span class="cm-operator">=</span> <span class="cm-variable">df</span>[<span class="cm-number">50</span>:<span class="cm-number">100</span>]</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">df2</span> <span class="cm-operator">=</span> <span class="cm-variable">df</span>[<span class="cm-number">100</span>:]</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 62px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 269px; margin-bottom: 0px; border-right-width: 30px; min-height: 45px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-keyword">import</span> <span class="cm-variable">matplotlib</span>.<span class="cm-property">pyplot</span> <span class="cm-keyword">as</span> <span class="cm-variable">plt</span></pre>
<pre class=" CodeMirror-line "><span class="cm-operator">%</span><span class="cm-variable">matplotlib</span> <span class="cm-variable">inline</span></pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 45px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell text_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="cell text_cell rendered unselected" tabindex="2">
<div class="inner_cell">
<div class="text_cell_render rendered_html" tabindex="-1">
<p><strong>Sepal length vs Sepal Width (Setosa vs Versicolor)</strong></p>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 739.4px; margin-bottom: 0px; border-right-width: 30px; min-height: 79px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-cursors">&nbsp;</div>
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">plt</span>.<span class="cm-property">xlabel</span>(<span class="cm-string">'Sepal Length'</span>)</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">plt</span>.<span class="cm-property">ylabel</span>(<span class="cm-string">'Sepal Width'</span>)</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">plt</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">df0</span>[<span class="cm-string">'sepal length (cm)'</span>], <span class="cm-variable">df0</span>[<span class="cm-string">'sepal width (cm)'</span>],<span class="cm-variable">color</span><span class="cm-operator">=</span><span class="cm-string">"green"</span>,<span class="cm-variable">marker</span><span class="cm-operator">=</span><span class="cm-string">'+'</span>)</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">plt</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">df1</span>[<span class="cm-string">'sepal length (cm)'</span>], <span class="cm-variable">df1</span>[<span class="cm-string">'sepal width (cm)'</span>],<span class="cm-variable">color</span><span class="cm-operator">=</span><span class="cm-string">"blue"</span>,<span class="cm-variable">marker</span><span class="cm-operator">=</span><span class="cm-string">'.'</span>)</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 79px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="output_area">&nbsp;</div>
<div class="output_area">
<div class="output_subarea output_text output_result">
<pre>&lt;matplotlib.collections.PathCollection at 0x1a1774abd0&gt;</pre>
</div>
</div>
<div class="output_area">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_png"><img src="data:image/png;base64,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" alt="" /></div>
</div>
</div>
</div>
</div>
<div class="cell text_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="cell text_cell rendered unselected" tabindex="2">
<div class="inner_cell">
<div class="text_cell_render rendered_html" tabindex="-1">
<p><strong>Petal length vs Pepal Width (Setosa vs Versicolor)</strong></p>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="prompt_container">&nbsp;</div>
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 739.4px; margin-bottom: 0px; border-right-width: 30px; min-height: 79px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">plt</span>.<span class="cm-property">xlabel</span>(<span class="cm-string">'Petal Length'</span>)</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">plt</span>.<span class="cm-property">ylabel</span>(<span class="cm-string">'Petal Width'</span>)</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">plt</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">df0</span>[<span class="cm-string">'petal length (cm)'</span>], <span class="cm-variable">df0</span>[<span class="cm-string">'petal width (cm)'</span>],<span class="cm-variable">color</span><span class="cm-operator">=</span><span class="cm-string">"green"</span>,<span class="cm-variable">marker</span><span class="cm-operator">=</span><span class="cm-string">'+'</span>)</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">plt</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">df1</span>[<span class="cm-string">'petal length (cm)'</span>], <span class="cm-variable">df1</span>[<span class="cm-string">'petal width (cm)'</span>],<span class="cm-variable">color</span><span class="cm-operator">=</span><span class="cm-string">"blue"</span>,<span class="cm-variable">marker</span><span class="cm-operator">=</span><span class="cm-string">'.'</span>)</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 79px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="output_area">&nbsp;</div>
<div class="output_area">
<div class="output_subarea output_text output_result">
<pre>&lt;matplotlib.collections.PathCollection at 0x1a17891550&gt;</pre>
</div>
</div>
<div class="output_area">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_png"><img src="data:image/png;base64,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" alt="" /></div>
</div>
</div>
</div>
</div>
<div class="cell text_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="cell text_cell rendered unselected" tabindex="2">
<div class="inner_cell">
<div class="text_cell_render rendered_html" tabindex="-1">
<p><strong>Train Using Support Vector Machine (SVM)</strong></p>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 445.4px; margin-bottom: 0px; border-right-width: 30px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-cursors">&nbsp;</div>
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">model_selection</span> <span class="cm-keyword">import</span> <span class="cm-variable">train_test_split</span></pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 28px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 453.8px; margin-bottom: 0px; border-right-width: 30px; min-height: 45px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-cursors">&nbsp;</div>
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">X</span> <span class="cm-operator">=</span> <span class="cm-variable">df</span>.<span class="cm-property">drop</span>([<span class="cm-string">'target'</span>,<span class="cm-string">'flower_name'</span>], <span class="cm-variable">axis</span><span class="cm-operator">=</span><span class="cm-string">'columns'</span>)</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">y</span> <span class="cm-operator">=</span> <span class="cm-variable">df</span>.<span class="cm-property">target</span></pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 45px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 613.4px; margin-bottom: 0px; border-right-width: 30px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-cursors">&nbsp;</div>
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">X_train</span>, <span class="cm-variable">X_test</span>, <span class="cm-variable">y_train</span>, <span class="cm-variable">y_test</span> <span class="cm-operator">=</span> <span class="cm-variable">train_test_split</span>(<span class="cm-variable">X</span>, <span class="cm-variable">y</span>, <span class="cm-variable">test_size</span><span class="cm-operator">=</span><span class="cm-number">0.2</span>)</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 28px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 109.4px; margin-bottom: 0px; border-right-width: 30px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-cursors">&nbsp;</div>
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-builtin">len</span>(<span class="cm-variable">X_train</span>)</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 28px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_text output_result">
<pre>120</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="prompt_container">&nbsp;</div>
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 101px; margin-bottom: 0px; border-right-width: 30px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-builtin">len</span>(<span class="cm-variable">X_test</span>)</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 28px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_text output_result">
<pre>30</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="prompt_container">&nbsp;</div>
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.59998px; left: 5.60001px;">&nbsp;</div>
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 235.4px; margin-bottom: 0px; border-right-width: 30px; min-height: 45px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">svm</span> <span class="cm-keyword">import</span> <span class="cm-variable">SVC</span></pre>
<pre class=" CodeMirror-line "><span class="cm-variable">model</span> <span class="cm-operator">=</span> <span class="cm-variable">SVC</span>()</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 45px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 235.4px; margin-bottom: 0px; border-right-width: 30px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-cursors">&nbsp;</div>
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">model</span>.<span class="cm-property">fit</span>(<span class="cm-variable">X_train</span>, <span class="cm-variable">y_train</span>) </pre>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_text output_result">
<pre>SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="prompt_container">&nbsp;</div>
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 235.4px; margin-bottom: 0px; border-right-width: 30px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">model</span>.<span class="cm-property">score</span>(<span class="cm-variable">X_test</span>, <span class="cm-variable">y_test</span>)</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 28px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_area">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_text output_result">
<pre>1.0</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="prompt_container">&nbsp;</div>
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 294.2px; margin-bottom: 0px; border-right-width: 30px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">model</span>.<span class="cm-property">predict</span>([[<span class="cm-number">4.8</span>,<span class="cm-number">3.0</span>,<span class="cm-number">1.5</span>,<span class="cm-number">0.3</span>]])</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 28px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_text output_result">
<pre>array([0])</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell text_cell rendered unselected" tabindex="2">
<div class="inner_cell">
<div class="text_cell_render rendered_html" tabindex="-1">
<p><strong>Tune parameters</strong></p>
</div>
</div>
</div>
<div class="cell text_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="cell text_cell rendered unselected" tabindex="2">
<div class="inner_cell">
<div class="text_cell_render rendered_html" tabindex="-1">
<p><strong>1. Regularization (C)</strong></p>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 252.2px; margin-bottom: 0px; border-right-width: 30px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">model_C</span> <span class="cm-operator">=</span> <span class="cm-variable">SVC</span>(<span class="cm-variable">C</span><span class="cm-operator">=</span><span class="cm-number">1</span>)</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">model_C</span>.<span class="cm-property">fit</span>(<span class="cm-variable">X_train</span>, <span class="cm-variable">y_train</span>)</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">model_C</span>.<span class="cm-property">score</span>(<span class="cm-variable">X_test</span>, <span class="cm-variable">y_test</span>)</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 62px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<p><br />&nbsp;</p>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_text output_result">
<pre>1.0</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="prompt_container">&nbsp;</div>
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.59998px; left: 5.60001px;">&nbsp;</div>
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 252.2px; margin-bottom: 0px; border-right-width: 30px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-measure">&nbsp;</div>
<div class="CodeMirror-measure">&nbsp;</div>
<div style="position: relative; z-index: 1;">&nbsp;</div>
<div class="CodeMirror-cursors">&nbsp;</div>
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">model_C</span> <span class="cm-operator">=</span> <span class="cm-variable">SVC</span>(<span class="cm-variable">C</span><span class="cm-operator">=</span><span class="cm-number">10</span>)</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">model_C</span>.<span class="cm-property">fit</span>(<span class="cm-variable">X_train</span>, <span class="cm-variable">y_train</span>)</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">model_C</span>.<span class="cm-property">score</span>(<span class="cm-variable">X_test</span>, <span class="cm-variable">y_test</span>)</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 62px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_text output_result">
<pre>1.0</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell text_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="cell text_cell rendered unselected" tabindex="2">
<div class="inner_cell">
<div class="text_cell_render rendered_html" tabindex="-1">
<p><strong>2. Gamma</strong></p>
</div>
</div>
</div>
<div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.59998px; left: 5.60001px;">&nbsp; </div>
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 252.2px; margin-bottom: 0px; border-right-width: 30px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-measure">&nbsp;</div>
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">model_g</span> <span class="cm-operator">=</span> <span class="cm-variable">SVC</span>(<span class="cm-variable">gamma</span><span class="cm-operator">=</span><span class="cm-number">10</span>)</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">model_g</span>.<span class="cm-property">fit</span>(<span class="cm-variable">X_train</span>, <span class="cm-variable">y_train</span>)</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">model_g</span>.<span class="cm-property">score</span>(<span class="cm-variable">X_test</span>, <span class="cm-variable">y_test</span>)</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 62px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_text output_result">
<pre>0.9333333333333333</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell text_cell rendered unselected" tabindex="2">
<div class="inner_cell">
<div class="text_cell_render rendered_html" tabindex="-1">
<p><strong>3. Kernel</strong></p>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 361.4px; margin-bottom: 0px; border-right-width: 30px; min-height: 45px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">model_linear_kernal</span> <span class="cm-operator">=</span> <span class="cm-variable">SVC</span>(<span class="cm-variable">kernel</span><span class="cm-operator">=</span><span class="cm-string">'linear'</span>)</pre>
<pre class=" CodeMirror-line "><span class="cm-variable">model_linear_kernal</span>.<span class="cm-property">fit</span>(<span class="cm-variable">X_train</span>, <span class="cm-variable">y_train</span>)</pre>
</div>
</div>
</div>
</div>
</div>
<div style="position: absolute; height: 30px; width: 1px; border-bottom: 0px solid transparent; top: 45px;">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_area">&nbsp;</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_text output_result">
<pre>SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='linear', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">&nbsp;</div>
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1" draggable="true">
<div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 353px; margin-bottom: 0px; border-right-width: 30px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;">
<div style="position: relative; top: 0px;">
<div class="CodeMirror-lines">
<div style="position: relative; outline: currentcolor none medium;">
<div class="CodeMirror-code">
<pre class=" CodeMirror-line "><span class="cm-variable">model_linear_kernal</span>.<span class="cm-property">score</span>(<span class="cm-variable">X_test</span>, <span class="cm-variable">y_test</span>)</pre>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">&nbsp;</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_subarea output_text output_result">
<pre>1.0</pre>
<p>&nbsp;</p>
</div>
