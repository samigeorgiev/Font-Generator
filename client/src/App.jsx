import React, { Component } from 'react';

import { Route, Switch } from 'react-router-dom';

import Auth from 'pages/Auth';
import Home from 'pages/Home';
import Layout from 'components/Layout';

class App extends Component {
    state = {
        user: null
    };

    loginHandler = (userId, token) => {
        this.setState({
            user: {
                id: userId,
                token: token
            }
        });
    };

    logoutHandler = () => {

    };

    render() {
        return (
            <Layout
                isAuth={Boolean(this.state.user)}
                logout={this.state.user ? this.logoutHandler : null}
            >
                <Switch>
                    <Route path="/" exact>
                        <Home />
                    </Route>
                    {!this.state.user
                        ? <Route path="/auth">
                            <Auth login={this.loginHandler} />
                        </Route>
                        : null}
                </Switch>
            </Layout>
        );
    }
}

export default App;