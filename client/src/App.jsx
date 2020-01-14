import React, { Component } from 'react';

import { Route, Switch } from 'react-router-dom';

import Auth from 'pages/Auth';
import Home from 'pages/Home';
import Layout from 'components/Layout';

class App extends Component {
    state = {
        user: null
    };

    render() {
        return (
            <Layout isAuth={Boolean(this.state.user)}>
                <Switch>
                    <Route path="/" exact>
                        <Home />
                    </Route>
                    {!this.state.user
                        ? <Route path="/auth">
                            <Auth />
                        </Route>
                        : null}
                </Switch>
            </Layout>
        );
    }
}

export default App;