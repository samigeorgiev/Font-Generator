import React, {Component} from 'react';

import { Route, Switch } from 'react-router-dom';

import Auth from 'pages/Auth';
import Layout from 'components/Layout';

class App extends Component {
    state = {
        user: null
    };

    render() {
        return (
            <Layout>
                <Switch>
                    <Route path="/">
                        <Auth />
                    </Route>
                </Switch>
            </Layout>
        );
    }
}

export default App;